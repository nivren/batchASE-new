"""
Single-process, single-GPU Worker for batched structure optimization.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from ase.io import read

from ..neighbors import AtomsToGraphs
from ..utils import data_list_collater, ensure_directory
from ..potentials import create_backend
from ..relaxation import (
    OptimizableBatch,
    OptimizableUnitCellBatch,
    OptimizableFrechetCellBatch,
    get_optimizer_cls,
)

logger = logging.getLogger("batchase.engine.worker")


class Worker:
    """
    Executes batched relaxation across assigned files on a dedicated GPU device.
    """

    def __init__(
        self,
        files: List[str] | str | Path,
        device: str,
        batch_size: int = 4,
        max_steps: int = 100,
        fmax: float = 0.01,
        filter1: Optional[str] = None,
        filter2: Optional[str] = None,
        optimizer1: str = "BFGSFusedLS",
        optimizer2: str = "BFGSFusedLS",
        skip_second_stage: bool = False,
        scalar_pressure: float = 0.0006,
        molecule_single: int = 64,
        output_path: str = "./",
        model: str = "mace",
        use_fasteq: bool = False,
        cueq: bool = False,
        bfgs_cpu_thread: int = 1,
        worker_id: int = 0,
        **kwargs,
    ) -> None:
        if isinstance(files, (str, Path)):
            manifest_file = Path(files)
            if manifest_file.is_file():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    self.files = [line.strip() for line in f if line.strip()]
            else:
                self.files = [str(files)]
        else:
            self.files = list(files)

        self.device = device
        self.worker_id = int(kwargs.get("worker_id", worker_id))
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.fmax = fmax
        self.filter1 = filter1
        self.filter2 = filter2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.skip_second_stage = skip_second_stage
        self.scalar_pressure = scalar_pressure
        self.molecule_single = molecule_single
        self.output_path = os.path.abspath(output_path)
        self.model = model
        self.use_fasteq = use_fasteq
        self.cueq = cueq
        self.bfgs_cpu_thread = bfgs_cpu_thread
        self.use_profiler = kwargs.get("use_profiler", False)
        self.profiler_log_dir = kwargs.get("profiler_log_dir", None)
        self.profiler_schedule_config = kwargs.get("profiler_schedule_config", None)

        if str(device).startswith("cuda"):
            dev_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            torch.cuda.set_device(dev_idx)

    @property
    def worker_tag(self) -> str:
        return f"[W{self.worker_id:02d}@{self.device}]"

    def _get_density(self, atoms) -> float:
        """Compute crystal density in g/cm^3."""
        try:
            vol = atoms.get_volume()
            mass = sum(atoms.get_masses())
            return (mass / vol) * 1.66053906660
        except Exception:
            return 0.0

    def _run_stage(
        self,
        files: List[str],
        stage_name: str,
        filter_type: Optional[str],
        optimizer_name: str,
        scalar_pressure: float,
        backend,
        a2g: AtomsToGraphs,
    ) -> tuple[List[str], Dict[str, Any]]:
        """Execute one complete relaxation stage with true continuous batch replenishment."""
        logger.info(
            f"{self.worker_tag} Starting {stage_name} on {len(files)} files with {optimizer_name} (filter={filter_type})."
        )
        cif_dir = ensure_directory(os.path.join(self.output_path, f"cif_result_{stage_name}"))
        json_dir = ensure_directory(os.path.join(self.output_path, f"json_result_{stage_name}"))

        empty_metrics: Dict[str, Any] = {
            "structures": 0,
            "steps": 0,
            "elapsed_s": 0.0,
            "mace_s": 0.0,
            "opt_s": 0.0,
            "graph_s": 0.0,
            "io_s": 0.0,
            "peak_vram_gb": 0.0,
        }
        if not files:
            return [], empty_metrics

        # Initial batch loading
        cur_batch_path = files[: self.batch_size]
        indices_to_process = len(cur_batch_path)
        optimized_atoms = [read(p) for p in cur_batch_path]

        gbatch = data_list_collater([a2g.convert(a) for a in optimized_atoms]).to(self.device)
        if filter_type == "UnitCellFilter":
            obatch = OptimizableUnitCellBatch(
                batch=gbatch,
                backend=backend,
                scalar_pressure=scalar_pressure,
                dtype=torch.float64,
            )
            orig_cells = obatch.orig_cells.clone()
        elif filter_type == "FrechetCellFilter":
            obatch = OptimizableFrechetCellBatch(
                batch=gbatch,
                backend=backend,
                scalar_pressure=scalar_pressure,
                dtype=torch.float64,
            )
            orig_cells = obatch.orig_cells.clone()
        else:
            obatch = OptimizableBatch(
                batch=gbatch,
                backend=backend,
                dtype=torch.float64,
            )
            orig_cells = None

        opt_cls = get_optimizer_cls(optimizer_name)
        opt_kwargs = {
            "maxstep": 0.2,
            "early_stop": True,
            "device": self.device,
            "use_profiler": self.use_profiler,
            "profiler_log_dir": self.profiler_log_dir,
            "profiler_schedule_config": self.profiler_schedule_config,
        }
        if optimizer_name == "BFGS":
            opt_kwargs["alpha"] = 70.0
            opt_kwargs["bfgs_cpu_thread"] = self.bfgs_cpu_thread
        elif optimizer_name in ("BFGSFusedLS", "BFGSLineSearch"):
            opt_kwargs["alpha"] = 10.0

        batch_optimizer = opt_cls(obatch, **opt_kwargs)

        cur_batch_steps = [0] * len(cur_batch_path)
        cur_batch_times = [time.perf_counter()] * len(cur_batch_path)
        all_indices = []
        converged_atoms_count = 0
        stage_start = time.perf_counter()
        last_heartbeat = stage_start
        total_steps_in_stage = 0
        total_burst_time = 0.0
        stage_mace_start = backend.mace_time
        stage_graph_start = backend.graph_time
        output_cif_paths = []

        while converged_atoms_count < len(files):
            # Dynamic batch replenishment when structures finish
            if len(all_indices) > 0:
                restart_indices = [i for i in range(len(optimized_atoms)) if i not in all_indices]
                old_batch_indices = obatch.batch_indices

                optimized_atoms_new = [optimized_atoms[i] for i in restart_indices]
                cur_batch_path_new = [cur_batch_path[i] for i in restart_indices]
                cur_batch_steps_new = [cur_batch_steps[i] for i in restart_indices]
                cur_batch_times_new = [cur_batch_times[i] for i in restart_indices]

                # Fill empty slots up to batch_size
                num_needed = self.batch_size - len(optimized_atoms_new)
                new_paths = files[indices_to_process : indices_to_process + num_needed]
                indices_to_process += len(new_paths)

                for np in new_paths:
                    optimized_atoms_new.append(read(np))
                    cur_batch_path_new.append(np)
                    cur_batch_steps_new.append(0)
                    cur_batch_times_new.append(time.perf_counter())

                if not optimized_atoms_new:
                    break

                # Preserve reference unit cells for surviving structures
                if filter_type in ("UnitCellFilter", "FrechetCellFilter"):
                    orig_cells_new = torch.zeros(
                        [len(optimized_atoms_new), 3, 3],
                        device=self.device,
                        dtype=torch.float64,
                    )
                    for new_i, old_i in enumerate(restart_indices):
                        orig_cells_new[new_i] = orig_cells[old_i]

                optimized_atoms = optimized_atoms_new
                cur_batch_path = cur_batch_path_new
                cur_batch_steps = cur_batch_steps_new
                cur_batch_times = cur_batch_times_new

                graphs_list = [a2g.convert(a) for a in optimized_atoms]
                gbatch = data_list_collater(graphs_list).to(self.device)

                if filter_type == "UnitCellFilter":
                    obatch = OptimizableUnitCellBatch(
                        batch=gbatch,
                        backend=backend,
                        scalar_pressure=scalar_pressure,
                        dtype=torch.float64,
                    )
                    for new_i in range(len(restart_indices)):
                        obatch.orig_cells[new_i] = orig_cells_new[new_i]
                    orig_cells = obatch.orig_cells.clone()
                elif filter_type == "FrechetCellFilter":
                    obatch = OptimizableFrechetCellBatch(
                        batch=gbatch,
                        backend=backend,
                        scalar_pressure=scalar_pressure,
                        dtype=torch.float64,
                    )
                    for new_i in range(len(restart_indices)):
                        obatch.orig_cells[new_i] = orig_cells_new[new_i]
                    orig_cells = obatch.orig_cells.clone()
                else:
                    obatch = OptimizableBatch(
                        batch=gbatch,
                        backend=backend,
                        dtype=torch.float64,
                    )

                batch_optimizer.optimizable = obatch

            # Compute remaining steps
            current_max_steps = max(cur_batch_steps) if cur_batch_steps else 0
            remaining_steps = max(self.max_steps - current_max_steps, 1)

            # Continuous optimization: runs until at least one structure converges
            t_burst_start = time.perf_counter()
            if len(all_indices) > 0 and hasattr(batch_optimizer, "restart_from_earlystop"):
                converge_indices = batch_optimizer.run(
                    self.fmax,
                    remaining_steps,
                    is_restart_earlystop=True,
                    restart_indices=restart_indices,
                    old_batch_indices=old_batch_indices,
                )
            else:
                converge_indices = batch_optimizer.run(self.fmax, remaining_steps)
            t_burst_end = time.perf_counter()
            burst_duration = t_burst_end - t_burst_start
            total_burst_time += burst_duration

            if converge_indices is None:
                converge_indices = []

            burst_steps = batch_optimizer.nsteps
            total_steps_in_stage += burst_steps

            # Update step count and detect completed/over-step slots
            cur_batch_steps = [s + burst_steps for s in cur_batch_steps]
            over_maxstep_indices = [
                i for i, s in enumerate(cur_batch_steps) if s >= self.max_steps
            ]
            all_indices = list(set(converge_indices + over_maxstep_indices))

            # Retrieve latest atoms and energies
            optimized_atoms = obatch.get_atoms_list()
            raw_energies = obatch.get_potential_energies()
            if hasattr(raw_energies, "view"):
                energies_list = raw_energies.view(-1).tolist()
            elif isinstance(raw_energies, np.ndarray):
                energies_list = raw_energies.flatten().tolist()
            elif hasattr(raw_energies, "tolist"):
                energies_list = raw_energies.tolist()
            else:
                energies_list = list(raw_energies)

            try:
                max_forces_tensor = obatch.get_max_forces()
                if hasattr(max_forces_tensor, "tolist"):
                    max_forces_list = max_forces_tensor.detach().cpu().tolist()
                else:
                    max_forces_list = [float(f) for f in max_forces_tensor]
            except Exception:
                max_forces_list = [0.0] * len(cur_batch_path)

            cur_elapsed = max(time.perf_counter() - stage_start, 1e-6)
            cur_mace_time = max(0.0, backend.mace_time - stage_mace_start)
            cur_graph_time = max(0.0, backend.graph_time - stage_graph_start)
            cur_opt_time = max(0.0, total_burst_time - (cur_mace_time + cur_graph_time))
            cur_io_time = max(0.0, cur_elapsed - total_burst_time)

            mace_ratio = cur_mace_time / cur_elapsed
            opt_ratio = cur_opt_time / cur_elapsed
            other_ratio = max(0.0, 1.0 - mace_ratio - opt_ratio)

            # Save results for finished structures
            end_time = time.perf_counter()
            for idx in all_indices:
                runtime = max(end_time - cur_batch_times[idx], 1e-6)
                steps = cur_batch_steps[idx]
                rate = steps / runtime
                is_conv = idx in converge_indices
                conv_str = "YES" if is_conv else "NO"
                fmax_val = max_forces_list[idx] if idx < len(max_forces_list) else 0.0

                mace_s = runtime * mace_ratio
                opt_s = runtime * opt_ratio
                other_s = runtime * other_ratio
                mace_pct = mace_ratio * 100.0
                opt_pct = opt_ratio * 100.0
                other_pct = other_ratio * 100.0

                natoms = len(optimized_atoms[idx])
                num_mol = natoms / self.molecule_single if self.molecule_single > 0 else 1.0
                e_raw = energies_list[idx] if idx < len(energies_list) else 0.0
                if isinstance(e_raw, (list, tuple)):
                    e_raw = e_raw[0]
                e_val = float(e_raw)
                energy_per_mol = (e_val / num_mol) * 96.485 if num_mol > 0 else e_val
                density = self._get_density(optimized_atoms[idx])

                stem = Path(cur_batch_path[idx]).stem
                out_cif = os.path.join(cif_dir, f"{stem}.cif")
                out_json = os.path.join(json_dir, f"{stem}.json")

                optimized_atoms[idx].write(out_cif)
                output_cif_paths.append(out_cif)

                result_data = {
                    "file": stem,
                    "converged": is_conv,
                    "fmax": fmax_val,
                    "steps": steps,
                    "runtime": runtime,
                    "energy": energy_per_mol,
                    "density": density,
                    "mace_time": mace_s,
                    "opt_time": opt_s,
                    "other_time": other_s,
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2)

                logger.info(
                    f"{self.worker_tag} [{stage_name}] DONE {stem}: "
                    f"conv={conv_str} steps={steps} ({rate:.1f} st/s) fmax={fmax_val:.4f} t={runtime:.1f}s "
                    f"[mace:{mace_s:.1f}s({mace_pct:.1f}%) opt:{opt_s:.1f}s({opt_pct:.1f}%) other:{other_s:.1f}s({other_pct:.1f}%)]"
                )

            converged_atoms_count += len(all_indices)

            now = time.perf_counter()
            if (converged_atoms_count > 0 and converged_atoms_count % 20 == 0) or (now - last_heartbeat >= 30.0):
                stage_elapsed = max(now - stage_start, 1e-6)
                surviving_count = len(cur_batch_path) - len(all_indices)
                active_slots = surviving_count if converged_atoms_count < len(files) else 0
                worker_rate = total_steps_in_stage / stage_elapsed
                vram_gb = (
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    if str(self.device).startswith("cuda")
                    else 0.0
                )
                logger.info(
                    f"{self.worker_tag} [{stage_name}] progress: {converged_atoms_count}/{len(files)} done | "
                    f"active_slots: {active_slots}/{self.batch_size} | "
                    f"avg_rate: {worker_rate:.1f} st/s | peak_vram: {vram_gb:.2f} GB"
                )
                last_heartbeat = now

        stage_total_time = max(time.perf_counter() - stage_start, 1e-6)
        final_mace_time = max(0.0, backend.mace_time - stage_mace_start)
        final_graph_time = max(0.0, backend.graph_time - stage_graph_start)
        final_opt_time = max(0.0, total_burst_time - (final_mace_time + final_graph_time))
        final_io_time = max(0.0, stage_total_time - total_burst_time)
        stage_rate = total_steps_in_stage / stage_total_time
        vram_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if str(self.device).startswith("cuda")
            else 0.0
        )
        logger.info(
            f"{self.worker_tag} [{stage_name}] completed: {len(files)} structures in {stage_total_time:.2f}s "
            f"({total_steps_in_stage} steps, {stage_rate:.1f} st/s, peak_vram: {vram_gb:.2f} GB)"
        )
        stage_metrics = {
            "structures": len(files),
            "steps": total_steps_in_stage,
            "elapsed_s": stage_total_time,
            "mace_s": final_mace_time,
            "opt_s": final_opt_time,
            "graph_s": final_graph_time,
            "io_s": final_io_time,
            "peak_vram_gb": vram_gb,
        }
        return output_cif_paths, stage_metrics

    def run(self) -> None:
        """Run complete 2-stage or 1-stage relaxation pipeline."""
        logger.info(f"{self.worker_tag} Worker started with {len(self.files)} files.")
        a2g = AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64)
        backend = create_backend(
            self.model,
            device=self.device,
            enable_cueq=self.cueq,
            use_fasteq=self.use_fasteq,
        )

        worker_start_time = time.perf_counter()

        # Stage 1: Pressure relaxation
        s1_cifs, s1_metrics = self._run_stage(
            files=self.files,
            stage_name="press",
            filter_type=self.filter1,
            optimizer_name=self.optimizer1,
            scalar_pressure=self.scalar_pressure,
            backend=backend,
            a2g=a2g,
        )

        stages_dict = {"press": s1_metrics}

        # Stage 2: Final relaxation
        if not self.skip_second_stage and s1_cifs:
            s2_cifs, s2_metrics = self._run_stage(
                files=s1_cifs,
                stage_name="final",
                filter_type=self.filter2,
                optimizer_name=self.optimizer2,
                scalar_pressure=0.0,
                backend=backend,
                a2g=a2g,
            )
            stages_dict["final"] = s2_metrics

        total_elapsed = time.perf_counter() - worker_start_time
        vram_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if str(self.device).startswith("cuda")
            else 0.0
        )

        total_steps = sum(m.get("steps", 0) for m in stages_dict.values())
        total_mace = sum(m.get("mace_s", 0.0) for m in stages_dict.values())
        total_opt = sum(m.get("opt_s", 0.0) for m in stages_dict.values())
        total_graph = sum(m.get("graph_s", 0.0) for m in stages_dict.values())
        total_io = sum(m.get("io_s", 0.0) for m in stages_dict.values())

        worker_metrics = {
            "worker_id": self.worker_id,
            "device": str(self.device),
            "stages": stages_dict,
            "total_elapsed_s": total_elapsed,
            "total_steps": total_steps,
            "mace_s": total_mace,
            "opt_s": total_opt,
            "graph_s": total_graph,
            "io_s": total_io,
            "peak_vram_gb": vram_gb,
        }

        metrics_dir = os.path.join(self.output_path, "metrics")
        ensure_directory(metrics_dir)
        metric_file = os.path.join(metrics_dir, f"worker_{self.worker_id}.json")
        try:
            with open(metric_file, "w", encoding="utf-8") as f:
                json.dump(worker_metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write worker metrics: {e}")

        logger.info(f"{self.worker_tag} Worker finished all stages in {total_elapsed:.2f}s.")
