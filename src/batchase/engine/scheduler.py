"""
Multi-worker and multi-GPU Scheduler for batch structure relaxation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional
import torch.multiprocessing as mp

from .worker import Worker
from ..utils import ensure_directory

logger = logging.getLogger("batchase.engine.scheduler")


def _worker_process_target(kwargs):
    """Entry point for worker multiprocessing process."""
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

        ensure_directory(self.output_path)

    def run(self) -> None:
        """Partition files and execute worker processes."""
        start_time = time.perf_counter()
        logger.info(
            f"Scheduler starting: {len(self.files)} files, {self.num_workers} workers, devices={self.devices}"
        )

        # Distribute files evenly among workers
        file_chunks = [[] for _ in range(self.num_workers)]
        for i, file_path in enumerate(self.files):
            file_chunks[i % self.num_workers].append(file_path)

        processes = []
        for worker_id in range(self.num_workers):
            chunk = file_chunks[worker_id]
            if not chunk:
                continue
            device = self.devices[worker_id % len(self.devices)]
            worker_kwargs = {
                "files": chunk,
                "device": device,
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
            }

            p = mp.Process(target=_worker_process_target, args=(worker_kwargs,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        elapsed = time.perf_counter() - start_time
        logger.info(f"All worker processes completed. Total elapsed time: {elapsed:.2f}s")
