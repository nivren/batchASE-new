"""
Multi-worker and multi-GPU Scheduler for batch structure relaxation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional
import torch.multiprocessing as mp

from .worker import Worker
from ..utils import ensure_directory

logger = logging.getLogger("batchase.engine.scheduler")


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
        molecule_single: int = 64,
        bfgs_cpu_thread: int = 1,
        **kwargs,
    ) -> None:
        self.files = list(files)
        self.num_workers = max(1, num_workers)
        self.devices = devices or ["cuda:0"]
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.fmax = fmax
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
        self.molecule_single = molecule_single
        self.bfgs_cpu_thread = bfgs_cpu_thread

        self.bind_cores = kwargs.pop("bind_cores", None)
        self.cpu_masks = self._parse_bind_cores(self.bind_cores)

        self.profile = kwargs.pop("profile", "False")
        self.use_profiler = False
        self.profiler_schedule_config = {"wait": 48, "warmup": 1, "active": 1, "repeat": 1}
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

    def run(self) -> None:
        """Partition files and execute worker processes."""
        start_time = time.perf_counter()
        logger.info(
            f"Scheduler starting: {len(self.files)} files, {self.num_workers} workers, devices={self.devices}"
        )
        if self.cpu_masks is not None:
            logger.info(f"Custom core binding active for {len(self.cpu_masks)} workers.")

        # Distribute files evenly among workers
        file_chunks = [[] for _ in range(self.num_workers)]
        for i, file_path in enumerate(self.files):
            file_chunks[i % self.num_workers].append(file_path)

        manifest_dir = os.path.join(self.output_path, "manifests")
        processes = []
        for worker_id in range(self.num_workers):
            chunk = file_chunks[worker_id]
            if not chunk:
                continue

            # Protect against 64 KiB kernel pipe payload limit during mp.spawn
            if len(chunk) > 1000:
                ensure_directory(manifest_dir)
                shard_manifest = os.path.join(manifest_dir, f"worker_{worker_id}.manifest")
                with open(shard_manifest, "w", encoding="utf-8") as f:
                    f.write("\n".join(chunk) + "\n")
                files_payload = shard_manifest
            else:
                files_payload = chunk

            device = self.devices[worker_id % len(self.devices)]
            worker_kwargs = {
                "files": files_payload,
                "device": device,
                "worker_id": worker_id,
                "batch_size": self.batch_size,
                "max_steps": self.max_steps,
                "fmax": self.fmax,
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
                "use_profiler": self.use_profiler,
                "profiler_log_dir": self.profiler_log_dir,
                "profiler_schedule_config": self.profiler_schedule_config,
                **self.extra_kwargs,
            }
            if self.cpu_masks is not None:
                worker_kwargs["affinity_cores"] = self.cpu_masks[worker_id]

            ctx = mp.get_context("spawn")
            p = ctx.Process(target=_worker_process_target, args=(worker_kwargs,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self._write_summary_csv()

        elapsed = time.perf_counter() - start_time
        self._print_dashboard(elapsed)

        summary_csv = os.path.join(self.output_path, "summary_scheduler.csv")
        try:
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                f.write("elapsed_time,num_workers,batch_size\n")
                f.write(f"{elapsed},{self.num_workers},{self.batch_size}\n")
        except Exception as e:
            logger.warning(f"Failed to write summary_scheduler.csv: {e}")
        logger.info(f"All worker processes completed. Total elapsed time: {elapsed:.2f}s")

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

            s2_steps = int(s2_data.get("steps", 0))
            s2_time = float(s2_data.get("runtime", 0.0))
            s2_energy = float(s2_data.get("energy", 0.0))
            s2_density = float(s2_data.get("density", 0.0))

            records.append({
                "file": stem,
                "stage1_steps": s1_steps,
                "stage1_time": s1_time,
                "stage1_energy": s1_energy,
                "stage1_density": s1_density,
                "stage2_steps": s2_steps,
                "stage2_time": s2_time,
                "stage2_energy": s2_energy,
                "stage2_density": s2_density,
                "total_steps": s1_steps + s2_steps,
                "total_time": s1_time + s2_time,
            })

        if records:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "file",
                        "stage1_steps",
                        "stage1_time",
                        "stage1_energy",
                        "stage1_density",
                        "stage2_steps",
                        "stage2_time",
                        "stage2_energy",
                        "stage2_density",
                        "total_steps",
                        "total_time",
                    ],
                )
                writer.writeheader()
                writer.writerows(records)
            logger.info(f"Summary CSV generated: {csv_file}")
        self.summary_records = records

    def _print_dashboard(self, elapsed: float) -> None:
        """Aggregate worker metrics and output performance summary dashboard."""
        metrics_dir = os.path.join(self.output_path, "metrics")
        worker_files = sorted(Path(metrics_dir).glob("worker_*.json")) if os.path.exists(metrics_dir) else []

        worker_data = []
        for wf in worker_files:
            try:
                with open(wf, "r", encoding="utf-8") as f:
                    worker_data.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to read metric file {wf}: {e}")

        if not worker_data:
            return

        total_structures = len(self.files)

        # Structure-level steps aggregated from results_scheduler.csv
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

        s1_struct_steps = sum(int(r.get("stage1_steps", 0)) for r in records)
        s2_struct_steps = sum(int(r.get("stage2_steps", 0)) for r in records)
        tot_struct_steps = sum(int(r.get("total_steps", 0)) for r in records)
        avg_s1_struct = s1_struct_steps / max(len(records), 1)
        avg_s2_struct = s2_struct_steps / max(len(records), 1)

        # Stage 1 metrics
        s1_mace = sum(w.get("stages", {}).get("press", {}).get("mace_s", 0.0) for w in worker_data)
        s1_opt = sum(w.get("stages", {}).get("press", {}).get("opt_s", 0.0) for w in worker_data)
        s1_graph = sum(w.get("stages", {}).get("press", {}).get("graph_s", 0.0) for w in worker_data)
        s1_io = sum(w.get("stages", {}).get("press", {}).get("io_s", 0.0) for w in worker_data)
        s1_steps = sum(w.get("stages", {}).get("press", {}).get("steps", 0) for w in worker_data)
        s1_total_time = max(s1_mace + s1_opt + s1_graph + s1_io, 1e-6)

        # Stage 2 metrics
        s2_mace = sum(w.get("stages", {}).get("final", {}).get("mace_s", 0.0) for w in worker_data)
        s2_opt = sum(w.get("stages", {}).get("final", {}).get("opt_s", 0.0) for w in worker_data)
        s2_graph = sum(w.get("stages", {}).get("final", {}).get("graph_s", 0.0) for w in worker_data)
        s2_io = sum(w.get("stages", {}).get("final", {}).get("io_s", 0.0) for w in worker_data)
        s2_steps = sum(w.get("stages", {}).get("final", {}).get("steps", 0) for w in worker_data)
        s2_total_time = max(s2_mace + s2_opt + s2_graph + s2_io, 1e-6)

        # Overall totals
        total_mace = s1_mace + s2_mace
        total_opt = s1_opt + s2_opt
        total_graph = s1_graph + s2_graph
        total_io = s1_io + s2_io
        total_worker_time = max(total_mace + total_opt + total_graph + total_io, 1e-6)
        total_batch_steps = s1_steps + s2_steps

        mace_pct = (total_mace / total_worker_time) * 100.0
        opt_pct = (total_opt / total_worker_time) * 100.0
        graph_pct = (total_graph / total_worker_time) * 100.0
        io_pct = (total_io / total_worker_time) * 100.0
        s1_opt_pct = (s1_opt / total_worker_time) * 100.0
        s2_opt_pct = (s2_opt / total_worker_time) * 100.0

        cluster_batch_rate = total_batch_steps / max(elapsed, 1e-6)
        cluster_struct_rate = tot_struct_steps / max(elapsed, 1e-6)
        structs_per_min = (total_structures / max(elapsed, 1e-6)) * 60.0

        lines = [
            "",
            "=" * 96,
            "                          batchASE Performance Dashboard",
            "=" * 96,
            f" Total Wall Time   : {elapsed:.2f}s",
            f" Total Structures  : {total_structures}",
            f" Structure Steps   : {tot_struct_steps:,} steps (S1: {s1_struct_steps:,} | S2: {s2_struct_steps:,}) [Avg: {avg_s1_struct:.1f} S1 / {avg_s2_struct:.1f} S2 per struct]",
            f" Batch Iterations  : {total_batch_steps:,} GPU steps (S1: {s1_steps:,} | S2: {s2_steps:,})",
            f" Cluster Throughput: {cluster_batch_rate:.1f} batch-steps/s ({cluster_struct_rate:.1f} struct-steps/s | {structs_per_min:.1f} structs/min)",
            f" Active Devices    : {self.devices} ({self.num_workers} workers, batch_size={self.batch_size})",
            "-" * 96,
            " Component                     Stage 1 (Press)   Stage 2 (Final)   Total Worker Time    Share (%)",
            "-" * 96,
            f" MLIP (MACE) Inference         {s1_mace:>8.1f}s        {s2_mace:>8.1f}s          {total_mace:>8.1f}s        {mace_pct:>5.1f}%",
            f" Optimizer (BFGS)              {s1_opt:>8.1f}s        {s2_opt:>8.1f}s          {total_opt:>8.1f}s        {opt_pct:>5.1f}%",
            f"   ├─ S1: BFGSFusedLS + Cell   {s1_opt:>8.1f}s               -            {s1_opt:>8.1f}s        {s1_opt_pct:>5.1f}%",
            f"   └─ S2: Standard BFGS              -          {s2_opt:>8.1f}s           {s2_opt:>8.1f}s        {s2_opt_pct:>5.1f}%",
            f" Neighbor Graph (PBC)          {s1_graph:>8.1f}s        {s2_graph:>8.1f}s          {total_graph:>8.1f}s        {graph_pct:>5.1f}%",
            f" Replenish & I/O               {s1_io:>8.1f}s        {s2_io:>8.1f}s          {total_io:>8.1f}s        {io_pct:>5.1f}%",
            "-" * 96,
            f" Total Active Worker Time      {s1_total_time:>8.1f}s        {s2_total_time:>8.1f}s          {total_worker_time:>8.1f}s       100.0%",
            "-" * 96,
            " Worker    Device     Structs   S1 Steps  S2 Steps  Tot Steps    S1 Time   S2 Time  Tot Time  Peak VRAM",
            "-" * 96,
        ]

        for w in sorted(worker_data, key=lambda x: x.get("worker_id", 0)):
            wid = w.get("worker_id", 0)
            dev = w.get("device", "unknown")
            stages = w.get("stages", {})
            s1 = stages.get("press", {})
            s2 = stages.get("final", {})
            structs = s1.get("structures", 0)
            w_s1_steps = s1.get("steps", 0)
            w_s2_steps = s2.get("steps", 0)
            w_steps = w.get("total_steps", 0)
            s1_t = s1.get("elapsed_s", 0.0)
            s2_t = s2.get("elapsed_s", 0.0)
            tot_t = w.get("total_elapsed_s", 0.0)
            vram = w.get("peak_vram_gb", 0.0)
            lines.append(
                f" W{wid:02d}       {dev:<10s} {structs:>7d}  {w_s1_steps:>8d}  {w_s2_steps:>8d}  {w_steps:>9d}   {s1_t:>7.1f}s  {s2_t:>7.1f}s  {tot_t:>7.1f}s   {vram:>5.2f} GB"
            )

        lines.append("=" * 96)
        lines.append("")

        logger.info("\n".join(lines))
