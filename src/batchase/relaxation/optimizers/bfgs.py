"""
Unified batched BFGS optimizer with flexible linear algebra device dispatch (GPU CUDA/DCU streams vs. CPU thread pool).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Union
import torch

from ..optimizable import OptimizableBatch
from .base import BatchOptimizer

try:
    from torch_scatter import scatter
except ImportError:
    scatter = None

logger = logging.getLogger("batchase.optimizers.bfgs")


class BFGS(BatchOptimizer):
    """
    Unified batched BFGS optimizer for molecular and crystal structure relaxation.
    
    Supports dynamic linear algebra execution:
    - linalg_device="gpu" / "cuda": uses CUDA streams for GPU-parallel eigh.
    - linalg_device="cpu": offloads H updates and eigh to CPU thread pool (MKL/OpenBLAS),
      which is optimal for small crystal cells (3N <= 300) or on DCU systems.
    - linalg_device="auto": automatic choice based on hardware and problem size.
    """

    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.2,
        alpha: float = 70.0,
        early_stop: bool = False,
        linalg_device: str = "auto",
        linalg_threads: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(optimizable=optimizable_batch, maxstep=maxstep)
        self.alpha = alpha
        self.early_stop = early_stop
        self.linalg_device = linalg_device.lower()
        self.linalg_threads = linalg_threads

        if self.linalg_device == "auto":
            self.state_device = torch.device(self.optimizable.device)
        elif self.linalg_device == "cpu":
            self.state_device = torch.device("cpu")
        else:
            self.state_device = torch.device(self.linalg_device)

        self._cpu_executor = (
            ThreadPoolExecutor(max_workers=self.linalg_threads)
            if (self.state_device.type == "cpu" and self.linalg_threads > 1)
            else None
        )

        self.initialize()

    def initialize(self) -> None:
        """Initialize inverse Hessian approximations and reference tensors."""
        self.H = [None] * self.optimizable.batch_size
        self.pos0 = torch.zeros_like(
            self.optimizable.get_positions().reshape(-1),
            device=self.state_device,
            dtype=torch.float64,
        )
        self.forces0 = torch.zeros_like(self.pos0, device=self.state_device, dtype=torch.float64)

    def update_slots(self, keep_indices: List[int], num_new_slots: int) -> None:
        """
        Dynamically update Hessian and reference history when batch composition changes.
        """
        old_batch_indices = self.optimizable.batch_indices
        H_new = []
        pos0_new = torch.zeros_like(
            self.optimizable.get_positions().reshape(-1),
            device=self.state_device,
            dtype=torch.float64,
        )
        forces0_new = torch.zeros_like(pos0_new, device=self.state_device, dtype=torch.float64)

        for i, idx in enumerate(keep_indices):
            mask_old = (idx == old_batch_indices.repeat_interleave(3))
            mask = (i == self.optimizable.batch_indices.repeat_interleave(3))
            H_new.append(self.H[idx])
            pos0_new[mask] = self.pos0[mask_old]
            forces0_new[mask] = self.forces0[mask_old]

        for _ in range(num_new_slots):
            H_new.append(None)

        self.H = H_new
        self.pos0 = pos0_new
        self.forces0 = forces0_new

    def prepare_step(self, pos: torch.Tensor, forces: torch.Tensor):
        batch_size = len(self.H)
        cur_indices = self.optimizable.batch_indices.repeat_interleave(3)
        calc_indices = [
            i for i, need_update in enumerate(self.optimizable.update_mask) if need_update
        ]

        dpos_list = [None] * batch_size

        if self.state_device.type == "cpu":
            # Offloaded CPU execution
            forces_cpu = forces.to(self.state_device)
            if self._cpu_executor is not None and len(calc_indices) > 1:
                def _eigh_cpu(i):
                    omega, V = torch.linalg.eigh(self.H[i])
                    f_i = forces_cpu[cur_indices == i]
                    dpos_i = V @ ((f_i @ V) / torch.abs(omega))
                    return i, dpos_i.to(self.device)

                futures = [self._cpu_executor.submit(_eigh_cpu, i) for i in calc_indices]
                for f in futures:
                    i, dpos_i = f.result()
                    dpos_list[i] = dpos_i
            else:
                for i in calc_indices:
                    omega, V = torch.linalg.eigh(self.H[i])
                    f_i = forces_cpu[cur_indices == i]
                    dpos_i = V @ ((f_i @ V) / torch.abs(omega))
                    dpos_list[i] = dpos_i.to(self.device)
        else:
            # GPU stream-parallel execution
            streams = [torch.cuda.Stream() for _ in calc_indices]
            for i, stream in zip(calc_indices, streams):
                with torch.cuda.stream(stream):
                    omega, V = torch.linalg.eigh(self.H[i])
                    f_i = forces[cur_indices == i]
                    dpos_list[i] = (V @ (f_i.t() @ V / torch.abs(omega)).t())
            torch.cuda.current_stream().synchronize()
            for stream in streams:
                stream.synchronize()

        for i in range(batch_size):
            if not self.optimizable.update_mask[i]:
                dpos_list[i] = torch.zeros_like(forces[cur_indices == i])

        dpos = torch.cat(dpos_list)
        steplengths = torch.sqrt(scatter(dpos ** 2, cur_indices, dim=0, reduce="sum"))
        return dpos, steplengths

    def determine_step(self, dpos: torch.Tensor, steplengths: torch.Tensor) -> torch.Tensor:
        scale = steplengths / self.maxstep
        cur_indices = self.optimizable.batch_indices.repeat_interleave(3)
        scale_expanded = scale[cur_indices]
        mask = scale_expanded > 1
        dpos[mask] = dpos[mask] / scale_expanded[mask]
        return dpos

    def update(self, pos: torch.Tensor, forces: torch.Tensor) -> None:
        batch_size = len(self.H)
        cur_indices = self.optimizable.batch_indices.repeat_interleave(3)
        pos_s = pos.to(self.state_device)
        forces_s = forces.to(self.state_device)

        for i in range(batch_size):
            mask = (cur_indices == i)
            pos_i = pos_s[mask]
            forces_i = forces_s[mask]
            pos0_i = self.pos0[mask]
            forces0_i = self.forces0[mask]

            if self.H[i] is None:
                self.H[i] = (
                    torch.eye(pos_i.numel(), device=self.state_device, dtype=torch.float64)
                    * self.alpha
                )
                continue

            dpos_i = pos_i - pos0_i
            if torch.abs(dpos_i).max() < 1e-7:
                continue

            if not self.optimizable.update_mask[i]:
                continue

            dforces_i = forces_i - forces0_i
            dg_i = self.H[i] @ dpos_i
            a_i = torch.dot(dforces_i, dpos_i)
            b_i = torch.dot(dpos_i, dg_i)
            outer_force = torch.outer(dforces_i, dforces_i)
            outer_dg = torch.outer(dg_i, dg_i)
            self.H[i] = self.H[i] - outer_force / a_i - outer_dg / b_i

        self.pos0 = pos_s.clone()
        self.forces0 = forces_s.clone()

    def step(self, fmax: float = 0.01) -> None:
        forces = self.optimizable.get_forces(apply_constraint=True).to(dtype=torch.float64).reshape(-1)
        pos = self.optimizable.get_positions().to(dtype=torch.float64).reshape(-1)

        self.update(pos, forces)
        dpos, steplengths = self.prepare_step(pos, forces)
        dpos = self.determine_step(dpos, steplengths)
        self.optimizable.set_positions((pos + dpos).reshape(-1, 3))

    def run(
        self,
        fmax: float = 0.01,
        steps: int = 100,
        is_restart_earlystop: bool = False,
        restart_indices: Optional[List[int]] = None,
        old_batch_indices: Optional[torch.Tensor] = None,
    ) -> List[int]:
        self.fmax = fmax
        iteration = 0

        if is_restart_earlystop and restart_indices is not None:
            num_new = self.optimizable.batch_size - len(restart_indices)
            self.update_slots(restart_indices, num_new)

        max_forces = self.optimizable.get_max_forces(apply_constraint=True)
        while iteration < steps and not self.optimizable.converged(
            forces=None, fmax=self.fmax, max_forces=max_forces, f_upper_limit=1e25
        ):
            self.step(fmax=fmax)
            iteration += 1
            max_forces = self.optimizable.get_max_forces(apply_constraint=True)

        self.nsteps = iteration
        if self.early_stop:
            return self.optimizable.converge_indices_list
        return self.optimizable.converged(forces=None, fmax=self.fmax, max_forces=max_forces)
