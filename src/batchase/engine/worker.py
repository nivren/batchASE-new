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
    get_optimizer_cls,
)
from .slot_manager import SlotManager

logger = logging.getLogger("batchase.engine.worker")


class Worker:
    """
    Executes batched relaxation across assigned files on a dedicated GPU device.
    """

    def __init__(
        self,
        files: List[str],
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
        **kwargs,
    ) -> None:
        self.files = files
        self.device = device
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

        if str(device).startswith("cuda"):
            dev_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            torch.cuda.set_device(dev_idx)

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
    ) -> List[str]:
        """Execute one complete relaxation stage with dynamic slot replenishment."""
        logger.info(f"Starting {stage_name} on {len(files)} files with {optimizer_name} (filter={filter_type}).")
        cif_dir = ensure_directory(os.path.join(self.output_path, f"cif_result_{stage_name}"))
        json_dir = ensure_directory(os.path.join(self.output_path, f"json_result_{stage_name}"))

        slot_mgr = SlotManager(
            files=files,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            molecule_single=self.molecule_single,
        )
        slot_mgr.fill_initial_batch()

        opt_cls = get_optimizer_cls(optimizer_name)
        stage_start = time.perf_counter()
        output_cif_paths = []

        while not slot_mgr.is_empty:
            cur_atoms = slot_mgr.current_atoms_list()
            cur_paths = slot_mgr.current_paths()
            if not cur_atoms:
                break

            # Collate batch
            graphs = [a2g.convert(atoms) for atoms in cur_atoms]
            gbatch = data_list_collater(graphs).to(self.device)

            # Build optimizable batch
            if filter_type == "UnitCellFilter":
                obatch = OptimizableUnitCellBatch(
                    batch=gbatch,
                    backend=backend,
                    scalar_pressure=scalar_pressure,
                    dtype=torch.float64,
                )
            else:
                obatch = OptimizableBatch(
                    batch=gbatch,
                    backend=backend,
                    dtype=torch.float64,
                )

            optimizer = opt_cls(
                optimizable_batch=obatch,
                maxstep=0.2,
                early_stop=True,
            )

            # Advance relaxation burst (up to 15 steps per replenishment checkpoint)
            burst_steps = min(15, self.max_steps)
            converged_indices = optimizer.run(fmax=self.fmax, steps=burst_steps)
            if converged_indices is None:
                converged_indices = []

            # Retrieve updated atoms and energies
            optimized_atoms = obatch.get_atoms_list()
            energies = obatch.get_potential_energies().tolist()

            keep_indices, num_new, finished = slot_mgr.update_and_replenish(
                converge_indices=converged_indices,
                elapsed_steps=optimizer.nsteps,
                latest_atoms=optimized_atoms,
                energies=energies,
            )

            # Save finished records
            for record in finished:
                out_cif = os.path.join(cif_dir, f"{record['id']}.cif")
                out_json = os.path.join(json_dir, f"{record['id']}.json")
                record["atoms"].write(out_cif)
                output_cif_paths.append(out_cif)
                record["density"] = self._get_density(record["atoms"])
                save_data = {k: v for k, v in record.items() if k != "atoms"}
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2)

                logger.info(
                    f"[{stage_name}] DONE {record['id']}: steps={record['steps']} "
                    f"time={record['runtime']:.1f}s E={record['energy']:.4f} eV/mol"
                )

        logger.info(f"{stage_name} finished in {time.perf_counter() - stage_start:.2f}s.")
        return output_cif_paths

    def run(self) -> None:
        """Run complete 2-stage or 1-stage relaxation pipeline."""
        logger.info(f"Worker process started on device {self.device} with {len(self.files)} files.")
        a2g = AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64)
        backend = create_backend(
            self.model,
            device=self.device,
            enable_cueq=self.cueq,
            use_fasteq=self.use_fasteq,
        )

        # Stage 1: Pressure relaxation
        s1_cifs = self._run_stage(
            files=self.files,
            stage_name="press",
            filter_type=self.filter1,
            optimizer_name=self.optimizer1,
            scalar_pressure=self.scalar_pressure,
            backend=backend,
            a2g=a2g,
        )

        # Stage 2: Final relaxation
        if not self.skip_second_stage and s1_cifs:
            self._run_stage(
                files=s1_cifs,
                stage_name="final",
                filter_type=self.filter2,
                optimizer_name=self.optimizer2,
                scalar_pressure=0.0,
                backend=backend,
                a2g=a2g,
            )

        logger.info("Worker execution completed successfully.")
