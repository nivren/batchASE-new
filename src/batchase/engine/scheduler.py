"""
Multi-worker and multi-GPU Scheduler for batch structure relaxation.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import uuid
from statistics import fmean, median
import time
from pathlib import Path
from typing import List, Optional
import torch.multiprocessing as mp

from .worker import Worker
from .batching import count_structure_atoms, describe_batch, select_structure_files
from .batch_cache import load_or_build_batch_plan
from ..utils import ensure_directory

logger = logging.getLogger("batchase.engine.scheduler")


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_float(value, positive: bool = False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0.0):
        return None
    return number


def _stage_attempted(record: dict, stage: str) -> bool:
    status = record.get(f"{stage}_status")
    status = str(status).strip().lower() if status is not None else ""
    return bool(status) or _safe_int(record.get(f"{stage}_steps")) > 0


def _summary_stats(values: list[float]):
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": fmean(values),
        "max": max(values),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _convergence_text(converged: int, attempted: int) -> str:
    rate = (converged / attempted * 100.0) if attempted else 0.0
    return f"{converged}/{attempted} ({rate:.1f}%)"


def _format_worker_tail(stage_metrics: dict) -> str:
    steps = _safe_int(stage_metrics.get("max_structure_steps"))
    structure = str(stage_metrics.get("max_structure_file") or "-")
    return f"{steps}/{structure}"


def _worker_process_target(kwargs):
    """Entry point for worker multiprocessing process."""
    import warnings
    warnings.filterwarnings("ignore", message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
    warnings.filterwarnings("ignore", message=".*is_fx_tracing will return true.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")
    warnings.filterwarnings("ignore", category=UserWarning, module="mace")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)
    affinity_cores = kwargs.pop("affinity_cores", None)
    if affinity_cores is not None:
        try:
            os.sched_setaffinity(0, affinity_cores)
            logger.info(f"Worker {os.getpid()} bound to physical cores: {sorted(affinity_cores)}")
        except Exception as e:
            logger.warning(f"Worker {os.getpid()} failed to bind cores: {e}")
    worker = Worker(**kwargs)
    worker.run()


class Scheduler:
    """
    Coordinates multi-process and multi-GPU batch relaxation across input CIF crystal structures.
    """

    def __init__(
        self,
        files: List[str],
        num_workers: int = 1,
        devices: Optional[List[str]] = None,
        batch_size: int = 4,
        max_steps: int = 100,
        fmax: float = 0.01,
        fmax1: Optional[float] = None,
        fmax2: Optional[float] = None,
        filter1: Optional[str] = "UnitCellFilter",
        filter2: Optional[str] = None,
        optimizer1: str = "BFGSFusedLS",
        optimizer2: str = "BFGSFusedLS",
        skip_second_stage: bool = False,
        scalar_pressure: float = 0.0006,
        output_path: str = "./",
        model: str = "mace",
        use_fasteq: bool = False,
        cueq: bool = False,
        molecule_single: Optional[int] = None,
        bfgs_cpu_thread: int = 1,
        compile_mode: Optional[str] = None,
        batch_mode: str = "bsize",
        max_batch_atoms: int = 0,
        structure_order: str = "rand",
        structure_order_seed: int = 42,
        fixed_batch_plan: Optional[str] = None,
        batch_plan_cache_dir: Optional[str] = None,
        batch_plan_cache_limit: int = 20,
        num_structures: int = 0,
        **kwargs,
    ) -> None:
        self._candidate_files = list(files)
        self.files = list(self._candidate_files)
        self.num_workers = max(1, num_workers)
        self.devices = devices or ["cuda:0"]
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.fmax = fmax
        self.fmax1 = fmax1 if fmax1 is not None else fmax
        self.fmax2 = fmax2 if fmax2 is not None else fmax
        self.filter1 = filter1
        self.filter2 = filter2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.skip_second_stage = skip_second_stage
        self.scalar_pressure = scalar_pressure
        self.output_path = os.path.abspath(output_path)
        self.model = model
        self.use_fasteq = use_fasteq
        self.cueq = cueq
        mol_single_raw = molecule_single if molecule_single is not None else kwargs.pop("molecule_single", None)
        if mol_single_raw is None:
            self.molecule_single = None
        else:
            try:
                parsed_molecule_single = int(mol_single_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("molecule_single must be a positive integer or None") from exc
            if parsed_molecule_single <= 0:
                raise ValueError("molecule_single must be a positive integer or None")
            self.molecule_single = parsed_molecule_single
        self.bfgs_cpu_thread = bfgs_cpu_thread
        self.compile_mode = compile_mode or kwargs.pop("compile_mode", None)
        self.batch_mode = batch_mode
        self.max_batch_atoms = int(max_batch_atoms)
        self.structure_order = structure_order
        self.structure_order_seed = int(structure_order_seed)
        if not isinstance(num_structures, int) or num_structures < 0:
            raise ValueError("num_structures must be a nonnegative integer")
        self.num_structures = num_structures
        self.batch_plan_cache_dir = batch_plan_cache_dir
        self.batch_plan_cache_limit = batch_plan_cache_limit
        # Strict one-round mode: one precomputed batch per worker, with no
        # refill queue.  The selector may reduce the input set so this plan
        # fits the atom budget while retaining exactly num_workers batches.
        self.fixed_batch_plan = fixed_batch_plan

        self.bind_cores = kwargs.pop("bind_cores", None)
        self.cpu_masks = self._parse_bind_cores(self.bind_cores)

        self.profile = kwargs.pop("profile", "False")
        self.use_profiler = False
        self.profiler_schedule_config = {"wait": 0, "warmup": 0, "active": 1, "repeat": 1}
        self.profiler_log_dir = None
        if self.profile and str(self.profile).lower() != "false":
            self.use_profiler = True
            self.profiler_log_dir = os.path.join(self.output_path, "log")
            ensure_directory(self.profiler_log_dir)
            if str(self.profile).lower() != "true":
                try:
                    cfg = json.loads(self.profile)
                    if isinstance(cfg, dict):
                        self.profiler_schedule_config.update(cfg)
                except Exception:
                    pass

        self.extra_kwargs = kwargs
        ensure_directory(self.output_path)

    def _parse_bind_cores(self, bind_cores: Optional[str]) -> Optional[List[set[int]]]:
        if not bind_cores:
            return None
        ranges = bind_cores.split(",")
        if len(ranges) != self.num_workers:
            logger.warning(
                f"bind_cores count ({len(ranges)}) does not match num_workers ({self.num_workers}). Ignoring core binding."
            )
            return None
        bindings = []
        for r in ranges:
            try:
                start_str, end_str = r.split("-")
                start = int(start_str.strip())
                end = int(end_str.strip())
                bindings.append(set(range(start, end + 1)))
            except Exception as e:
                logger.warning(f"Failed to parse bind_cores element '{r}': {e}")
                return None
        return bindings

    def _load_fixed_batch_plan(self):
        """Load and validate an exact one-batch-per-worker plan."""
        if not self.fixed_batch_plan:
            raise ValueError("fixed_batch_plan is empty")
        plan_path = os.path.abspath(os.fspath(self.fixed_batch_plan))
        with open(plan_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_workers = raw.get("workers") if isinstance(raw, dict) else raw
        if not isinstance(raw_workers, list) or len(raw_workers) != self.num_workers:
            raise ValueError(
                f"fixed_batch_plan must contain exactly {self.num_workers} workers; "
                f"got {len(raw_workers) if isinstance(raw_workers, list) else type(raw_workers).__name__}"
            )

        assignments = []
        for worker_id, item in enumerate(raw_workers):
            files = item.get("files") if isinstance(item, dict) else item
            if not isinstance(files, list) or not files:
                raise ValueError(f"fixed_batch_plan worker {worker_id} has no files")
            assignments.append([os.path.abspath(os.fspath(path)) for path in files])

        expected = {os.path.abspath(os.fspath(path)) for path in self.files}
        actual = [path for batch in assignments for path in batch]
        actual_set = set(actual)
        if len(actual) != len(actual_set):
            raise ValueError("fixed_batch_plan contains duplicate input files")
        if actual_set != expected:
            missing = sorted(expected - actual_set)
            extra = sorted(actual_set - expected)
            raise ValueError(
                "fixed_batch_plan file union does not match scheduler inputs "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

        worker_atoms = []
        for worker_id, files in enumerate(assignments):
            atom_total = sum(count_structure_atoms(path) for path in files)
            worker_atoms.append(atom_total)
            if self.max_batch_atoms and atom_total > self.max_batch_atoms:
                raise ValueError(
                    f"fixed_batch_plan worker {worker_id} exceeds atom budget: "
                    f"{atom_total} > {self.max_batch_atoms}"
                )
        return assignments, worker_atoms, raw

    def _worker_kwargs(self, worker_id, files, **batch_kwargs):
        kwargs = {
            "files": files,
            "device": self.devices[worker_id % len(self.devices)],
            "worker_id": worker_id,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
            "max_batch_atoms": self.max_batch_atoms,
            "structure_order": self.structure_order,
            "structure_order_seed": self.structure_order_seed,
            "max_steps": self.max_steps,
            "fmax": self.fmax,
            "fmax1": self.fmax1,
            "fmax2": self.fmax2,
            "filter1": self.filter1,
            "filter2": self.filter2,
            "optimizer1": self.optimizer1,
            "optimizer2": self.optimizer2,
            "skip_second_stage": self.skip_second_stage,
            "scalar_pressure": self.scalar_pressure,
            "molecule_single": self.molecule_single,
            "output_path": self.output_path,
            "model": self.model,
            "use_fasteq": self.use_fasteq,
            "cueq": self.cueq,
            "bfgs_cpu_thread": self.bfgs_cpu_thread,
            "compile_mode": self.compile_mode,
            "use_profiler": self.use_profiler,
            "profiler_log_dir": self.profiler_log_dir,
            "profiler_schedule_config": self.profiler_schedule_config,
            **self.extra_kwargs,
            **batch_kwargs,
        }
        if self.cpu_masks is not None:
            kwargs["affinity_cores"] = self.cpu_masks[worker_id]
        return kwargs

    def _execute_workers(self, ctx, workers):
        """Stop promptly on a process failure instead of reporting success."""
        processes = []
        try:
            for kwargs in workers:
                process = ctx.Process(target=_worker_process_target, args=(kwargs,))
                process.start()
                processes.append(process)
            while any(process.is_alive() for process in processes):
                for process in processes:
                    process.join(timeout=0.1)
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"Worker process {process.pid} failed with exit code {process.exitcode}"
                        )
            for process in processes:
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"Worker process {process.pid} failed with exit code {process.exitcode}"
                    )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()

    def _verify_batch_completion(self, plan):
        """A successful run must have a current completion record for every batch."""
        for batch in plan["batches"]:
            path = Path(self.output_path) / "metrics" / f"batch_{batch['batch_id']}.json"
            try:
                with path.open(encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Missing completion record for batch {batch['batch_id']}") from exc
            if (record.get("run_id") != plan["run_id"]
                    or record.get("status") != "completed"
                    or record.get("files") != batch["files"]
                    or record.get("stages", {}).get("press", {}).get("structures") != batch["nfiles"]):
                raise RuntimeError(f"Invalid completion record for batch {batch['batch_id']}")

    def run(self) -> None:
        """Plan complete batches, then execute queued or externally fixed work."""
        self.files = list(self._candidate_files)
        start_time = time.perf_counter()
        run_id = uuid.uuid4().hex
        self.run_id = run_id
        ctx = mp.get_context("spawn")
        batch_queue = None
        manifest_dir = Path(self.output_path) / "manifests"
        ensure_directory(manifest_dir)
        plan_path = Path(self.output_path) / "batch_plan.json"

        if self.fixed_batch_plan:
            # External exact plans retain worker ownership and file order.
            if self.num_structures:
                self.files = select_structure_files(
                    self.files, self.structure_order, self.structure_order_seed, self.num_structures)
            assignments, worker_atoms, _ = self._load_fixed_batch_plan()
            counts = {path: count_structure_atoms(path) for files in assignments for path in files}
            plan = {
                "mode": "exact-one-batch-per-worker",
                "batch_mode": "fixed",
                "source_plan": os.path.abspath(os.fspath(self.fixed_batch_plan)),
                "num_files": len(self.files),
                "num_workers": self.num_workers,
                "num_batches": self.num_workers,
                "batch_size_limit": self.batch_size,
                "max_batch_atoms": self.max_batch_atoms,
                "worker_atom_totals": worker_atoms,
                "batches": [describe_batch(files, counts, worker_id, self.max_batch_atoms)
                            for worker_id, files in enumerate(assignments)],
                "workers": [{"worker_id": index, "num_batches": 1,
                             "num_files": len(files), "atom_total": worker_atoms[index],
                             "files": files} for index, files in enumerate(assignments)],
            }
            workers = []
            for worker_id, files in enumerate(assignments):
                # Preserve the existing manifests while keeping spawn payloads small.
                with (manifest_dir / f"worker_{worker_id}.batches.json").open("w", encoding="utf-8") as handle:
                    json.dump([files], handle, indent=2)
                payload = files
                if len(files) > 1000:
                    shard = manifest_dir / f"worker_{worker_id}.manifest"
                    shard.write_text("\n".join(files) + "\n", encoding="utf-8")
                    payload = str(shard)
                workers.append(self._worker_kwargs(
                    worker_id, payload, fixed_one_batch=True, run_id=run_id,
                    batch_plan_path=str(plan_path),
                ))
        else:
            plan, cache_report = load_or_build_batch_plan(
                self.files, self.batch_mode, self.batch_size, self.max_batch_atoms,
                self.structure_order, self.structure_order_seed,
                cache_dir=self.batch_plan_cache_dir,
                cache_limit=self.batch_plan_cache_limit,
                num_structures=self.num_structures,
            )
            plan["planning_cache"] = cache_report
            logger.info("Batch plan cache %s: %s (planning %.3fs)",
                        cache_report["status"], cache_report["path"] or "-",
                        cache_report["elapsed_s"])
            self.files = plan["ordered_files"]
            active_workers = min(self.num_workers, plan["num_batches"])
            plan["num_workers"] = self.num_workers
            plan["active_workers"] = active_workers
            if active_workers:
                batch_queue = ctx.Queue()
                # Queue only IDs: CIF data and graph tensors stay out of IPC.
                for batch in plan["batches"]:
                    batch_queue.put(batch["batch_id"])
                for _ in range(active_workers):
                    batch_queue.put(None)
            workers = [self._worker_kwargs(
                worker_id, [], batch_queue=batch_queue, batch_plan_path=str(plan_path),
                run_id=run_id,
            ) for worker_id in range(active_workers)]
        plan["run_id"] = run_id
        with plan_path.open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        (Path(self.output_path) / "manifest.txt").write_text(
            "\n".join(self.files) + ("\n" if self.files else ""), encoding="utf-8",
        )
        logger.info(
            f"Batch plan: mode={plan['batch_mode']}, files={len(self.files)}, "
            f"batches={plan['num_batches']}, active_workers={len(workers)}; "
            f"devices={self.devices}."
        )
        try:
            self._execute_workers(ctx, workers)
            self._verify_batch_completion(plan)
        finally:
            if batch_queue is not None:
                # A crashed worker may leave unread tasks; never wait for the
                # feeder to flush those tasks into a pipe with no readers.
                batch_queue.cancel_join_thread()
                batch_queue.close()

        self._write_summary_csv()
        elapsed = time.perf_counter() - start_time
        self._print_dashboard(elapsed)
        summary_csv = os.path.join(self.output_path, "summary_scheduler.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_time", "num_workers", "batch_size"])
            writer.writerow([elapsed, self.num_workers, self.batch_size])
        logger.info(f"All batches completed. Total elapsed time: {elapsed:.2f}s")

    def _write_summary_csv(self) -> None:
        """Aggregate per-structure JSON records into results_scheduler.csv."""
        press_dir = os.path.join(self.output_path, "json_result_press")
        final_dir = os.path.join(self.output_path, "json_result_final")
        csv_file = os.path.join(self.output_path, "results_scheduler.csv")

        records = []
        for file_path in self.files:
            stem = Path(file_path).stem
            press_json = os.path.join(press_dir, f"{stem}.json")
            final_json = os.path.join(final_dir, f"{stem}.json")

            s1_data = {}
            if os.path.exists(press_json):
                try:
                    with open(press_json, "r", encoding="utf-8") as f:
                        s1_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read {press_json}: {e}")

            s2_data = {}
            if os.path.exists(final_json):
                try:
                    with open(final_json, "r", encoding="utf-8") as f:
                        s2_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read {final_json}: {e}")

            s1_steps = int(s1_data.get("steps", 0))
            s1_time = float(s1_data.get("runtime", 0.0))
            s1_energy = float(s1_data.get("energy", 0.0))
            s1_density = float(s1_data.get("density", 0.0))
            s1_fmax = s1_data.get("fmax")
            s1_fmax_atom = s1_data.get("fmax_atom")
            s1_fmax_stress = s1_data.get("fmax_stress")
            s1_fmax_stress_gpa = s1_data.get("fmax_stress_gpa")
            s2_steps = int(s2_data.get("steps", 0))
            s2_time = float(s2_data.get("runtime", 0.0))
            s2_energy = float(s2_data.get("energy", 0.0))
            s2_density = float(s2_data.get("density", 0.0))
            s2_fmax = s2_data.get("fmax")
            s2_fmax_atom = s2_data.get("fmax_atom")
            s2_fmax_stress = s2_data.get("fmax_stress")
            s2_fmax_stress_gpa = s2_data.get("fmax_stress_gpa")

            s1_status = s1_data.get("status", "converged" if s1_data.get("converged") else "failed")
            s1_failed_reason = s1_data.get("failed_reason") or ""
            s2_status = s2_data.get("status", "converged" if s2_data.get("converged") else "failed") if s2_data else ""
            s2_failed_reason = (s2_data.get("failed_reason") or "") if s2_data else ""

            if s2_data:
                final_status = s2_status
                final_failed_reason = s2_failed_reason
            else:
                final_status = s1_status
                final_failed_reason = s1_failed_reason

            natoms = int(s2_data.get("natoms") or s1_data.get("natoms") or 0)
            num_molecules = s2_data.get("num_molecules") if "num_molecules" in s2_data else s1_data.get("num_molecules")
            norm_status = s2_data.get("normalization_status") or s1_data.get("normalization_status") or "unnormalized"

            s1_energy_kj_mol = s1_data.get("energy_kj_mol")
            s1_enthalpy_kj_mol = s1_data.get("enthalpy_kj_mol")
            s2_energy_kj_mol = s2_data.get("energy_kj_mol")
            s2_enthalpy_kj_mol = s2_data.get("enthalpy_kj_mol")

            records.append({
                "file": stem,
                "status": final_status,
                "failed_reason": final_failed_reason,
                "natoms": natoms,
                "num_molecules": num_molecules if num_molecules is not None else "",
                "normalization_status": norm_status,
                "stage1_status": s1_status,
                "stage1_failed_reason": s1_failed_reason,
                "stage1_steps": s1_steps,
                "stage1_time": s1_time,
                "stage1_energy": s1_energy,
                "stage1_energy_kj_mol": s1_energy_kj_mol if s1_energy_kj_mol is not None else "",
                "stage1_enthalpy_kj_mol": s1_enthalpy_kj_mol if s1_enthalpy_kj_mol is not None else "",
                "stage1_density": s1_density,
                "stage1_fmax": s1_fmax if s1_fmax is not None else "",
                "stage1_fmax_atom": s1_fmax_atom if s1_fmax_atom is not None else "",
                "stage1_fmax_stress": s1_fmax_stress if s1_fmax_stress is not None else "",
                "stage1_fmax_stress_gpa": s1_fmax_stress_gpa if s1_fmax_stress_gpa is not None else "",
                "stage2_status": s2_status,
                "stage2_failed_reason": s2_failed_reason,
                "stage2_steps": s2_steps,
                "stage2_time": s2_time,
                "stage2_energy": s2_energy,
                "stage2_energy_kj_mol": s2_energy_kj_mol if s2_energy_kj_mol is not None else "",
                "stage2_enthalpy_kj_mol": s2_enthalpy_kj_mol if s2_enthalpy_kj_mol is not None else "",
                "stage2_density": s2_density,
                "stage2_fmax": s2_fmax if s2_fmax is not None else "",
                "stage2_fmax_atom": s2_fmax_atom if s2_fmax_atom is not None else "",
                "stage2_fmax_stress": s2_fmax_stress if s2_fmax_stress is not None else "",
                "stage2_fmax_stress_gpa": s2_fmax_stress_gpa if s2_fmax_stress_gpa is not None else "",
                "total_steps": s1_steps + s2_steps,
                "total_time": s1_time + s2_time,
            })

        if records:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "file",
                        "status",
                        "failed_reason",
                        "natoms",
                        "num_molecules",
                        "normalization_status",
                        "stage1_status",
                        "stage1_failed_reason",
                        "stage1_steps",
                        "stage1_time",
                        "stage1_energy",
                        "stage1_energy_kj_mol",
                        "stage1_enthalpy_kj_mol",
                        "stage1_density",
                        "stage1_fmax",
                        "stage1_fmax_atom",
                        "stage1_fmax_stress",
                        "stage1_fmax_stress_gpa",
                        "stage2_status",
                        "stage2_failed_reason",
                        "stage2_steps",
                        "stage2_time",
                        "stage2_energy",
                        "stage2_energy_kj_mol",
                        "stage2_enthalpy_kj_mol",
                        "stage2_density",
                        "stage2_fmax",
                        "stage2_fmax_atom",
                        "stage2_fmax_stress",
                        "stage2_fmax_stress_gpa",
                        "total_steps",
                        "total_time",
                    ],
                )
                writer.writeheader()
                writer.writerows(records)
            logger.info(f"Summary CSV generated: {csv_file}")
        self.summary_records = records

    def _build_final_structure_summary(self, records: list[dict], has_stage2: bool) -> list[str]:
        stage = "stage2" if has_stage2 else "stage1"
        stage_label = "Stage 2 (Final)" if has_stage2 else "Stage 1 only"
        attempted = [record for record in records if _stage_attempted(record, stage)]
        converged = [
            record
            for record in attempted
            if str(record.get(f"{stage}_status", "")).strip().lower() == "converged"
        ]

        density_values = []
        fmax_values = []
        atom_fmax_values = []
        stress_gpa_values = []
        for record in converged:
            density = _finite_float(record.get(f"{stage}_density"), positive=True)
            if density is not None:
                density_values.append(density)
            fmax = _finite_float(record.get(f"{stage}_fmax"))
            if fmax is not None and fmax >= 0.0:
                fmax_values.append(fmax)
            atom_fmax = _finite_float(record.get(f"{stage}_fmax_atom"))
            if atom_fmax is not None and atom_fmax >= 0.0:
                atom_fmax_values.append(atom_fmax)
            stress_gpa = _finite_float(record.get(f"{stage}_fmax_stress_gpa"))
            if stress_gpa is not None and stress_gpa >= 0.0:
                stress_gpa_values.append(stress_gpa)

        pressure = _finite_float(self.scalar_pressure) or 0.0
        if has_stage2 or abs(pressure) <= 0.0:
            energy_key = f"{stage}_energy_kj_mol"
            energy_label = "Energy"
        else:
            energy_key = f"{stage}_enthalpy_kj_mol"
            energy_label = "Enthalpy"

        energy_values = []
        for record in converged:
            normalization_status = str(record.get("normalization_status", "")).strip().lower()
            if normalization_status in {"unnormalized", "invalid_atom_count"}:
                continue
            energy = _finite_float(record.get(energy_key))
            if energy is not None:
                energy_values.append(energy)

        density_stats = _summary_stats(density_values)
        fmax_stats = _summary_stats(fmax_values)
        atom_fmax_stats = _summary_stats(atom_fmax_values)
        stress_gpa_stats = _summary_stats(stress_gpa_values)
        energy_stats = _summary_stats(energy_values)
        stage1_attempted = [record for record in records if _stage_attempted(record, "stage1")]
        stage1_converged = sum(
            str(record.get("stage1_status", "")).strip().lower() == "converged"
            for record in stage1_attempted
        )
        stage2_attempted = [record for record in records if _stage_attempted(record, "stage2")]
        stage2_converged = sum(
            str(record.get("stage2_status", "")).strip().lower() == "converged"
            for record in stage2_attempted
        )
        lines = [
            " Convergence Rates",
            f"  S1                   : {_convergence_text(stage1_converged, len(stage1_attempted))}",
        ]
        if has_stage2:
            lines.append(f"  S2                   : {_convergence_text(stage2_converged, len(stage2_attempted))}")
        lines.extend([
            f"  Final                : {_convergence_text(len(converged), len(attempted))}",
            " Final Structure Summary",
            f"  Final stage          : {stage_label}",
            f"  Attempted / converged: {len(attempted)} / {len(converged)} (failed: {len(attempted) - len(converged)})",
        ])

        if fmax_stats is None:
            lines.append("  Final fmax [eV/A]    : unavailable")
        else:
            target_fmax = self.fmax2 if has_stage2 else self.fmax1
            lines.append(
                f"  Final fmax [eV/A]    : n={fmax_stats['count']} "
                f"median={fmax_stats['median']:.6f} p95={_percentile(fmax_values, 0.95):.6f} "
                f"max={fmax_stats['max']:.6f} below_target={sum(value <= target_fmax for value in fmax_values)}/{len(fmax_values)} "
                f"(target={target_fmax:.6f})"
            )

        if atom_fmax_stats is None:
            lines.append("  Final atom fmax [eV/A]: unavailable")
        else:
            lines.append(
                f"  Final atom fmax [eV/A]: n={atom_fmax_stats['count']} "
                f"median={atom_fmax_stats['median']:.6f} p95={_percentile(atom_fmax_values, 0.95):.6f} "
                f"max={atom_fmax_stats['max']:.6f}"
            )

        if stress_gpa_stats is None:
            lines.append("  Final stress [GPa]   : unavailable")
        else:
            lines.append(
                f"  Final stress [GPa]   : n={stress_gpa_stats['count']} "
                f"median={stress_gpa_stats['median']:.6f} p95={_percentile(stress_gpa_values, 0.95):.6f} "
                f"max={stress_gpa_stats['max']:.6f}"
            )

        if density_stats is None:
            lines.append("  Density [g/cm^3]     : unavailable")
        else:
            lines.append(
                f"  Density [g/cm^3]     : n={density_stats['count']} "
                f"min={density_stats['min']:.4f} median={density_stats['median']:.4f} "
                f"mean={density_stats['mean']:.4f} max={density_stats['max']:.4f}"
            )

        if energy_stats is None:
            lines.append(f"  {energy_label} [kJ/mol per molecule] : unavailable (no normalized finite values)")
        else:
            minimum = energy_stats["min"]
            deltas = [energy - minimum for energy in energy_values]
            lines.append(
                f"  {energy_label} [kJ/mol per molecule] : n={energy_stats['count']} "
                f"min={energy_stats['min']:.4f} median={energy_stats['median']:.4f} "
                f"mean={energy_stats['mean']:.4f} max={energy_stats['max']:.4f}"
            )
            lines.append(
                f"  Relative {energy_label:<9}: median_delta={median(deltas):.4f} "
                f"max_delta={max(deltas):.4f} within_5={sum(delta <= 5.0 for delta in deltas)}/{len(deltas)}"
            )
        return lines

    def _print_dashboard(self, elapsed: float) -> None:
        """Aggregate worker metrics and output performance summary dashboard."""
        metrics_dir = os.path.join(self.output_path, "metrics")
        worker_files = sorted(Path(metrics_dir).glob("worker_*.json")) if os.path.exists(metrics_dir) else []

        worker_data = []
        for wf in worker_files:
            try:
                with open(wf, "r", encoding="utf-8") as f:
                    record = json.load(f)
                if getattr(self, "run_id", None) is None or record.get("run_id") == self.run_id:
                    worker_data.append(record)
            except Exception as e:
                logger.warning(f"Failed to read metric file {wf}: {e}")

        if not worker_data:
            return

        total_structures = len(self.files)
        records = getattr(self, "summary_records", None)
        if not records:
            csv_file = os.path.join(self.output_path, "results_scheduler.csv")
            if os.path.exists(csv_file):
                try:
                    with open(csv_file, "r", encoding="utf-8") as f:
                        records = list(csv.DictReader(f))
                except Exception:
                    records = []
            else:
                records = []

        stage1_records = [record for record in records if _stage_attempted(record, "stage1")]
        stage2_records = [record for record in records if _stage_attempted(record, "stage2")]
        has_stage2 = bool(stage2_records)
        if not records:
            has_stage2 = any("final" in worker.get("stages", {}) for worker in worker_data)

        s1_struct_steps = sum(_safe_int(record.get("stage1_steps")) for record in stage1_records)
        s2_struct_steps = sum(_safe_int(record.get("stage2_steps")) for record in stage2_records)
        tot_struct_steps = s1_struct_steps + s2_struct_steps
        avg_s1_struct = s1_struct_steps / max(len(stage1_records), 1)
        avg_s2_struct = s2_struct_steps / max(len(stage2_records), 1)

        s1_mace = sum(w.get("stages", {}).get("press", {}).get("mace_s", 0.0) for w in worker_data)
        s1_opt = sum(w.get("stages", {}).get("press", {}).get("opt_s", 0.0) for w in worker_data)
        s1_graph = sum(w.get("stages", {}).get("press", {}).get("graph_s", 0.0) for w in worker_data)
        s1_io = sum(w.get("stages", {}).get("press", {}).get("io_s", 0.0) for w in worker_data)
        s1_total_time = max(s1_mace + s1_opt + s1_graph + s1_io, 1e-6)

        s2_mace = sum(w.get("stages", {}).get("final", {}).get("mace_s", 0.0) for w in worker_data)
        s2_opt = sum(w.get("stages", {}).get("final", {}).get("opt_s", 0.0) for w in worker_data)
        s2_graph = sum(w.get("stages", {}).get("final", {}).get("graph_s", 0.0) for w in worker_data)
        s2_io = sum(w.get("stages", {}).get("final", {}).get("io_s", 0.0) for w in worker_data)
        s2_total_time = max(s2_mace + s2_opt + s2_graph + s2_io, 1e-6)

        total_mace = s1_mace + s2_mace
        total_opt = s1_opt + s2_opt
        total_graph = s1_graph + s2_graph
        total_io = s1_io + s2_io
        total_worker_time = max(total_mace + total_opt + total_graph + total_io, 1e-6)

        mace_pct = (total_mace / total_worker_time) * 100.0
        opt_pct = (total_opt / total_worker_time) * 100.0
        graph_pct = (total_graph / total_worker_time) * 100.0
        io_pct = (total_io / total_worker_time) * 100.0
        s1_opt_pct = (s1_opt / total_worker_time) * 100.0
        s2_opt_pct = (s2_opt / total_worker_time) * 100.0

        cluster_struct_rate = tot_struct_steps / max(elapsed, 1e-6)
        structs_per_min = (total_structures / max(elapsed, 1e-6)) * 60.0

        model_name = self.model.upper() if hasattr(self, "model") and self.model else "MACE"
        c_mlip = f" MLIP ({model_name}) Inference"[:31].ljust(31)
        opt_header = (
            f"Optimizer ({self.optimizer1})"
            if self.optimizer1 == self.optimizer2
            else f"Optimizer ({self.optimizer1}/{self.optimizer2})"
        )
        c_opt = f" {opt_header}"[:31].ljust(31)
        filt1 = (" + Cell" if "Cell" in str(self.filter1) else f" + {self.filter1}") if self.filter1 else ""
        s1_tag = f"S1: {self.optimizer1}{filt1}"
        c_s1 = f"   ├─ {s1_tag}"[:31].ljust(31)

        lines = [
            "",
            "=" * 96,
            "                          batchASE Performance Dashboard",
            "=" * 96,
            f" Total Wall Time   : {elapsed:.2f}s",
            f" Total Structures  : {total_structures}",
        ]
        if has_stage2:
            lines.extend([
                f" Per-Structure Steps: {tot_struct_steps:,} steps (S1: {s1_struct_steps:,} | S2: {s2_struct_steps:,}) [Avg: {avg_s1_struct:.1f} S1 / {avg_s2_struct:.1f} S2 per attempted struct]",
            ])
        else:
            lines.extend([
                f" Per-Structure Steps: {s1_struct_steps:,} steps (S1: {s1_struct_steps:,}) [Avg: {avg_s1_struct:.1f} S1 per attempted struct]",
            ])
        lines.extend([
            f" Cluster Throughput: {cluster_struct_rate:.1f} struct-steps/s ({structs_per_min:.1f} structs/min)",
            f" Active Devices    : {self.devices} ({len(worker_data)} active workers)",
            f" Batch Planning    : {'fixed' if self.fixed_batch_plan else self.batch_mode}",
        ])
        lines.extend(self._build_final_structure_summary(records, has_stage2))
        lines.extend(["-" * 96])

        if has_stage2:
            lines.extend([
                " Component                     Stage 1 (Press)   Stage 2 (Final)   Total Worker Time    Share (%)",
                "-" * 96,
                f"{c_mlip}{s1_mace:>8.1f}s        {s2_mace:>8.1f}s          {total_mace:>8.1f}s        {mace_pct:>5.1f}%",
                f"{c_opt}{s1_opt:>8.1f}s        {s2_opt:>8.1f}s          {total_opt:>8.1f}s        {opt_pct:>5.1f}%",
                f"{c_s1}{s1_opt:>8.1f}s               -            {s1_opt:>8.1f}s        {s1_opt_pct:>5.1f}%",
                f"   └─ S2: {self.optimizer2}{(' + Cell' if 'Cell' in str(self.filter2) else f' + {self.filter2}') if self.filter2 else ''}"[:31].ljust(31) + f"      -          {s2_opt:>8.1f}s           {s2_opt:>8.1f}s        {s2_opt_pct:>5.1f}%",
                f" Neighbor Graph (PBC)          {s1_graph:>8.1f}s        {s2_graph:>8.1f}s          {total_graph:>8.1f}s        {graph_pct:>5.1f}%",
                f" Data & I/O                    {s1_io:>8.1f}s        {s2_io:>8.1f}s          {total_io:>8.1f}s        {io_pct:>5.1f}%",
                "-" * 96,
                f" Total Active Worker Time      {s1_total_time:>8.1f}s        {s2_total_time:>8.1f}s          {total_worker_time:>8.1f}s       100.0%",
                "-" * 96,
                " Worker    Device     Structs   S1 Time   S2 Time  Tot Time  Peak VRAM  S1 Max (steps/id)       S2 Max (steps/id)",
                "-" * 96,
            ])
        else:
            lines.extend([
                " Component                     Stage 1 (Press)   Total Worker Time    Share (%)",
                "-" * 96,
                f"{c_mlip}{s1_mace:>8.1f}s          {total_mace:>8.1f}s        {mace_pct:>5.1f}%",
                f"{c_opt}{s1_opt:>8.1f}s          {total_opt:>8.1f}s        {opt_pct:>5.1f}%",
                f"{c_s1}{s1_opt:>8.1f}s          {s1_opt:>8.1f}s        {s1_opt_pct:>5.1f}%",
                f" Neighbor Graph (PBC)          {s1_graph:>8.1f}s          {total_graph:>8.1f}s        {graph_pct:>5.1f}%",
                f" Data & I/O                    {s1_io:>8.1f}s          {total_io:>8.1f}s        {io_pct:>5.1f}%",
                "-" * 96,
                f" Total Active Worker Time      {s1_total_time:>8.1f}s          {total_worker_time:>8.1f}s       100.0%",
                "-" * 96,
                " Worker    Device     Structs   S1 Time  Tot Time  Peak VRAM  S1 Max (steps/id)",
                "-" * 96,
            ])

        for worker in sorted(worker_data, key=lambda item: item.get("worker_id", 0)):
            worker_id = worker.get("worker_id", 0)
            device = worker.get("device", "unknown")
            stages = worker.get("stages", {})
            stage1 = stages.get("press", {})
            stage2 = stages.get("final", {})
            structures = stage1.get("structures", 0)
            stage1_time = stage1.get("elapsed_s", 0.0)
            stage2_time = stage2.get("elapsed_s", 0.0)
            total_time = worker.get("total_elapsed_s", 0.0)
            vram = worker.get("peak_vram_gb", 0.0)
            stage1_tail = _format_worker_tail(stage1)
            stage2_tail = _format_worker_tail(stage2)
            if has_stage2:
                lines.append(
                    f" W{worker_id:02d}       {device:<10s} {structures:>7d}  {stage1_time:>7.1f}s  {stage2_time:>7.1f}s  {total_time:>7.1f}s   {vram:>5.2f} GB  {stage1_tail:<25s} {stage2_tail:<25s}"
                )
            else:
                lines.append(
                    f" W{worker_id:02d}       {device:<10s} {structures:>7d}  {stage1_time:>7.1f}s  {total_time:>7.1f}s   {vram:>5.2f} GB  {stage1_tail}"
                )

        lines.append("=" * 96)
        lines.append("")
        logger.info("\n".join(lines))
