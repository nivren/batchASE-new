"""
Unified batched BFGS optimizer with GPU CUDA stream parallelism and CPU thread pool execution.
Matches legacy batchASE numerical and slot replenishment semantics with machine precision.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Union
import torch

from ..optimizable import OptimizableBatch
from .base import BatchOptimizer

try:
    from torch_scatter import scatter
    _has_scatter = True
except ImportError:
    scatter = None
    _has_scatter = False

logger = logging.getLogger("batchase.optimizers.bfgs")


class BFGS(BatchOptimizer):
    """
    Batched BFGS optimizer for molecular and crystal structure relaxation.
    
    Automatically dispatches between GPU stream execution and CPU thread pool execution
    based on `bfgs_cpu_thread` / `linalg_device`.
    """

    def __new__(
        cls,
        optimizable_batch: OptimizableBatch,
        *args,
        bfgs_cpu_thread: int = 0,
        linalg_device: str = "auto",
        **kwargs,
    ):
        if cls is not BFGS:
            return super().__new__(cls)

        use_cpu = (bfgs_cpu_thread > 0) or (linalg_device.lower() == "cpu")
        if use_cpu:
            return _BFGSCpu(optimizable_batch, *args, bfgs_cpu_thread=bfgs_cpu_thread, **kwargs)
        return _BFGSGpu(optimizable_batch, *args, **kwargs)


class _BFGSGpu(BFGS):
    """
    GPU CUDA stream parallel batched BFGS optimizer.
    """

    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.2,
        alpha: float = 70.0,
        early_stop: bool = False,
        **kwargs,
    ) -> None:
        super(BFGS, self).__init__(optimizable=optimizable_batch, maxstep=maxstep)
        self.alpha = alpha
        self.early_stop = early_stop
        self.device = torch.device(self.optimizable.device)
        self.state_device = self.device
        self._streams = []
        self.initialize()

    def initialize(self) -> None:
        batch_size = self.optimizable.batch_size
        self.H = [None] * batch_size
        pos = self.optimizable.get_positions()
        self.pos0 = torch.zeros_like(pos.reshape(-1), device=self.device, dtype=torch.float64)
        self.forces0 = torch.zeros_like(self.pos0, device=self.device, dtype=torch.float64)
        if self.device.type == "cuda":
            self._streams = [torch.cuda.Stream(device=self.device) for _ in range(batch_size)]

    def restart_from_earlystop(self, restart_indices: List[int], old_batch_indices: torch.Tensor) -> None:
        batch_size = self.optimizable.batch_size
        H_new = []
        pos0_new = torch.zeros_like(
            self.optimizable.get_positions().reshape(-1),
            device=self.device,
            dtype=torch.float64,
        )
        forces0_new = torch.zeros_like(pos0_new, device=self.device, dtype=torch.float64)

        old_flat = old_batch_indices.repeat_interleave(3)
        new_flat = self.optimizable.batch_indices.repeat_interleave(3)

        for i, idx in enumerate(restart_indices):
            mask_old = (idx == old_flat)
            mask_new = (i == new_flat)
            H_new.append(self.H[idx] if idx < len(self.H) else None)
            pos0_new[mask_new] = self.pos0[mask_old]
            forces0_new[mask_new] = self.forces0[mask_old]

        for _ in range(len(H_new), batch_size):
            H_new.append(None)

        self.H = H_new
        self.pos0 = pos0_new
        self.forces0 = forces0_new

        if self.device.type == "cuda":
            while len(self._streams) < batch_size:
                self._streams.append(torch.cuda.Stream(device=self.device))

    def update_slots(self, keep_indices: List[int], num_new_slots: int) -> None:
        """Alias for compatibility with slot management interface."""
        old_batch_indices = self.optimizable.batch_indices
        self.restart_from_earlystop(keep_indices, old_batch_indices)

    def run(
        self,
        fmax: float = 0.01,
        steps: int = 100,
        is_restart_earlystop: bool = False,
        restart_indices: Optional[List[int]] = None,
        old_batch_indices: Optional[torch.Tensor] = None,
    ) -> List[int]:
        self.fmax = fmax
        self.max_iter = steps

        if is_restart_earlystop and restart_indices is not None and old_batch_indices is not None:
            self.restart_from_earlystop(restart_indices, old_batch_indices)

        iteration = 0
        max_forces = self.optimizable.get_max_forces(apply_constraint=True)

        while iteration < self.max_iter and not self.optimizable.converged(
            forces=None, fmax=self.fmax, max_forces=max_forces, f_upper_limit=1e25
        ):
            if self.early_stop and iteration > 0:
                converge_indices = self.optimizable.converge_indices_list
                if len(converge_indices) > 0:
                    break

            self.step()
            max_forces = self.optimizable.get_max_forces(apply_constraint=True)
            iteration += 1

        self.nsteps = iteration
        if self.early_stop:
            return self.optimizable.converge_indices_list
        return self.optimizable.converged(forces=None, fmax=self.fmax, max_forces=max_forces)

    def step(self, fmax: float = 0.01) -> None:
        forces = self.optimizable.get_forces(apply_constraint=True).to(dtype=torch.float64)
        pos = self.optimizable.get_positions().to(dtype=torch.float64)
        dpos, steplengths = self.prepare_step(pos, forces)
        dpos = self.determine_step(dpos, steplengths)
        self.optimizable.set_positions(pos + dpos)

    def prepare_step(self, pos: torch.Tensor, forces: torch.Tensor):
        forces_flat = forces.reshape(-1)
        pos_flat = pos.reshape(-1)
        self.update(pos_flat, forces_flat, self.pos0, self.forces0)

        cur_indices = self.optimizable.batch_indices.repeat_interleave(3)
        calc_indices = [
            i for i, need_update in enumerate(self.optimizable.update_mask) if need_update
        ]

        dpos_list = [None] * len(self.H)

        if self.device.type == "cuda" and len(self._streams) >= len(self.H):
            for i in calc_indices:
                stream = self._streams[i]
                with torch.cuda.stream(stream):
                    omega, V = torch.linalg.eigh(self.H[i])
                    omega_abs = torch.clamp(torch.abs(omega), min=1e-6)
                    f_i = forces_flat[cur_indices == i]
                    dpos_list[i] = (V @ (f_i.t() @ V / omega_abs).t())

            torch.cuda.current_stream().synchronize()
            for i in calc_indices:
                self._streams[i].synchronize()
        else:
            for i in calc_indices:
                omega, V = torch.linalg.eigh(self.H[i])
                omega_abs = torch.clamp(torch.abs(omega), min=1e-6)
                f_i = forces_flat[cur_indices == i]
                dpos_list[i] = (V @ (f_i.t() @ V / omega_abs).t())

        for i in range(len(self.H)):
            if not self.optimizable.update_mask[i]:
                dpos_list[i] = torch.zeros_like(forces_flat[cur_indices == i])

        dpos = torch.zeros_like(forces_flat)
        for i in torch.unique(cur_indices):
            mask = (cur_indices == i)
            dpos[mask] = dpos_list[i]
        dpos = dpos.reshape(-1, 3)

        steplengths = (dpos ** 2).sum(dim=-1).sqrt()
        self.pos0 = pos_flat
        self.forces0 = forces_flat
        return dpos, steplengths

    def determine_step(self, dpos: torch.Tensor, steplengths: torch.Tensor) -> torch.Tensor:
        if _has_scatter and dpos.is_cuda:
            longest_steps = scatter(
                steplengths, self.optimizable.batch_indices, reduce="max"
            )
        else:
            index = self.optimizable.batch_indices
            src = steplengths
            num_groups = int(index.max().item()) + 1 if index.numel() > 0 else 0
            out = torch.full((num_groups,), float("-inf"), device=src.device, dtype=src.dtype)
            longest_steps = out.scatter_reduce(
                dim=0, index=index, src=src, reduce="amax", include_self=True
            )

        longest_steps = longest_steps[self.optimizable.batch_indices]
        maxstep = longest_steps.new_tensor(self.maxstep)
        scale = (longest_steps).reciprocal() * torch.min(longest_steps, maxstep)
        dpos *= scale.unsqueeze(1)
        return dpos

    def update(self, pos: torch.Tensor, forces: torch.Tensor, pos0: torch.Tensor, forces0: torch.Tensor) -> None:
        dpos = pos - pos0
        dforces = forces - forces0
        batch_indices_flatten = self.optimizable.batch_indices.repeat_interleave(3)
        dg = torch.zeros_like(dforces)
        all_size = self.optimizable.elem_per_group

        for i in range(self.optimizable.batch_size):
            if self.H[i] is None:
                continue
            mask = (i == batch_indices_flatten)
            if torch.abs(dpos[mask]).max() < 1e-7:
                continue
            dg[mask] = self.H[i] @ dpos[mask]

        a = self._batched_dot_1d(dforces, dpos)
        b = self._batched_dot_1d(dpos, dg)

        for i in range(self.optimizable.batch_size):
            if self.H[i] is None:
                self.H[i] = (
                    torch.eye(3 * all_size[i], device=self.device, dtype=torch.float64) * self.alpha
                )
                continue
            mask = (i == batch_indices_flatten)
            if not self.optimizable.update_mask[i]:
                continue
            if torch.abs(dpos[mask]).max() < 1e-7:
                continue
            if torch.abs(a[i]) < 1e-12 or torch.abs(b[i]) < 1e-12:
                continue

            outer_force = torch.outer(dforces[mask], dforces[mask])
            outer_dg = torch.outer(dg[mask], dg[mask])
            self.H[i] -= outer_force / a[i] + outer_dg / b[i]

    def _batched_dot_1d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        index = self.optimizable.batch_indices.repeat_interleave(3)
        if _has_scatter and x.is_cuda:
            return scatter(x * y, index, reduce="sum")
        else:
            src = x * y
            num_groups = int(index.max().item()) + 1 if index.numel() > 0 else 0
            out = torch.zeros(num_groups, device=src.device, dtype=src.dtype)
            out.scatter_add_(dim=0, index=index, src=src)
            return out


class _BFGSCpu(BFGS):
    """
    CPU-threaded batched BFGS optimizer for systems where CPU linear algebra is preferred.
    """

    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.2,
        alpha: float = 70.0,
        early_stop: bool = False,
        bfgs_cpu_thread: int = 16,
        **kwargs,
    ) -> None:
        super(BFGS, self).__init__(optimizable=optimizable_batch, maxstep=maxstep)
        self.alpha = alpha
        self.early_stop = early_stop
        self.device = torch.device(self.optimizable.device)
        self.state_device = torch.device("cpu")
        self._executor: Optional[ThreadPoolExecutor] = None

        self._set_thread_config(bfgs_cpu_thread)
        self._refresh_runtime_metadata()
        self.initialize()
        self._build_executor()

    def _set_thread_config(self, bfgs_cpu_thread: int) -> None:
        batch_size = max(1, int(self.optimizable.batch_size))
        self.bfgs_cpu_thread = max(1, min(int(bfgs_cpu_thread), batch_size))

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _build_executor(self) -> None:
        self._shutdown_executor()
        if self.bfgs_cpu_thread > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.bfgs_cpu_thread)

    def __del__(self):
        try:
            self._shutdown_executor()
        except Exception:
            pass

    def _refresh_runtime_metadata(self) -> None:
        self.batch_size = int(self.optimizable.batch_size)
        self.elem_per_group = [int(x) for x in self.optimizable.elem_per_group]
        self.coord_sizes = [3 * n for n in self.elem_per_group]

        flat_batch_indices = self.optimizable.batch_indices.detach().cpu().repeat_interleave(3)
        expected = torch.repeat_interleave(
            torch.arange(self.batch_size),
            torch.tensor(self.coord_sizes, dtype=torch.int64),
        )
        self._use_slices = bool(
            flat_batch_indices.numel() == expected.numel() and torch.equal(flat_batch_indices, expected)
        )

        self._coord_refs = []
        if self._use_slices:
            start = 0
            for size in self.coord_sizes:
                stop = start + size
                self._coord_refs.append(slice(start, stop))
                start = stop
        else:
            self._coord_refs = [
                torch.nonzero(flat_batch_indices == i, as_tuple=False).reshape(-1)
                for i in range(self.batch_size)
            ]
        self._set_thread_config(self.bfgs_cpu_thread)

    def _coord_view(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        ref = self._coord_refs[idx]
        return x[ref]

    def initialize(self) -> None:
        self.H = [None] * self.batch_size
        coord_dim = int(sum(self.coord_sizes))
        self.pos0 = torch.zeros(coord_dim, device=self.state_device, dtype=torch.float64)
        self.forces0 = torch.zeros_like(self.pos0)

    def restart_from_earlystop(self, restart_indices: List[int], old_batch_indices: torch.Tensor) -> None:
        self._refresh_runtime_metadata()
        self._build_executor()

        H_new = []
        pos0_new = torch.zeros(int(sum(self.coord_sizes)), device=self.state_device, dtype=torch.float64)
        forces0_new = torch.zeros_like(pos0_new)

        old_flat = old_batch_indices.detach().cpu().repeat_interleave(3)
        new_flat = self.optimizable.batch_indices.detach().cpu().repeat_interleave(3)

        for i, idx in enumerate(restart_indices):
            old_idx = int(idx)
            mask_old = old_flat == old_idx
            mask_new = new_flat == i
            H_new.append(self.H[old_idx] if old_idx < len(self.H) else None)
            pos0_new[mask_new] = self.pos0[mask_old]
            forces0_new[mask_new] = self.forces0[mask_old]

        for _ in range(len(H_new), self.batch_size):
            H_new.append(None)

        self.H = H_new
        self.pos0 = pos0_new
        self.forces0 = forces0_new

    def update_slots(self, keep_indices: List[int], num_new_slots: int) -> None:
        old_batch_indices = self.optimizable.batch_indices
        self.restart_from_earlystop(keep_indices, old_batch_indices)

    def run(
        self,
        fmax: float = 0.01,
        steps: int = 100,
        is_restart_earlystop: bool = False,
        restart_indices: Optional[List[int]] = None,
        old_batch_indices: Optional[torch.Tensor] = None,
    ) -> List[int]:
        self.fmax = fmax
        self.max_iter = steps

        if is_restart_earlystop and restart_indices is not None and old_batch_indices is not None:
            self.restart_from_earlystop(restart_indices, old_batch_indices)

        iteration = 0
        max_forces = self.optimizable.get_max_forces(apply_constraint=True)

        while iteration < self.max_iter and not self.optimizable.converged(
            forces=None, fmax=self.fmax, max_forces=max_forces, f_upper_limit=1e25
        ):
            if self.early_stop and iteration > 0:
                converge_indices = self.optimizable.converge_indices_list
                if len(converge_indices) > 0:
                    break

            self.step()
            max_forces = self.optimizable.get_max_forces(apply_constraint=True)
            iteration += 1

        self.nsteps = iteration
        if self.early_stop:
            return self.optimizable.converge_indices_list
        return self.optimizable.converged(forces=None, fmax=self.fmax, max_forces=max_forces)

    def step(self, fmax: float = 0.01) -> None:
        forces = self.optimizable.get_forces(apply_constraint=True).to(dtype=torch.float64)
        pos = self.optimizable.get_positions().to(dtype=torch.float64)
        dpos, steplengths = self.prepare_step(pos, forces)
        dpos = self.determine_step(dpos, steplengths)
        self.optimizable.set_positions(pos + dpos)

    @staticmethod
    def _eigh_single_cpu(idx: int, H_cpu: torch.Tensor, force_vec: torch.Tensor):
        omega, V = torch.linalg.eigh(H_cpu)
        omega_abs = torch.clamp(torch.abs(omega), min=1e-6)
        dpos_i = V @ ((force_vec @ V) / omega_abs)
        return idx, dpos_i

    def _update_single_cpu(
        self,
        idx: int,
        pos_i: torch.Tensor,
        forces_i: torch.Tensor,
        pos0_i: torch.Tensor,
        forces0_i: torch.Tensor,
        need_update: bool,
    ):
        H_i = self.H[idx]
        if H_i is None:
            H_i = torch.eye(pos_i.numel(), device=self.state_device, dtype=torch.float64) * self.alpha
            return idx, H_i

        dpos_i = pos_i - pos0_i
        if torch.abs(dpos_i).max() < 1e-7 or not need_update:
            return idx, H_i

        dforces_i = forces_i - forces0_i
        dg_i = H_i @ dpos_i
        a_i = torch.dot(dforces_i, dpos_i)
        b_i = torch.dot(dpos_i, dg_i)
        if torch.abs(a_i) > 1e-12 and torch.abs(b_i) > 1e-12:
            outer_force = torch.outer(dforces_i, dforces_i)
            outer_dg = torch.outer(dg_i, dg_i)
            H_i = H_i - outer_force / a_i - outer_dg / b_i
        return idx, H_i

    def prepare_step(self, pos: torch.Tensor, forces: torch.Tensor):
        pos_state = pos.view(-1).to(self.state_device)
        forces_state = forces.reshape(-1).to(self.state_device)
        self.update(pos_state, forces_state, self.pos0, self.forces0)

        dpos_state = torch.zeros_like(forces_state)
        calc_indices = [
            i for i, need_update in enumerate(self.optimizable.update_mask) if need_update
        ]

        if calc_indices:
            worker_args = [
                (i, self.H[i], self._coord_view(forces_state, i))
                for i in calc_indices
            ]
            if self._executor is not None and len(worker_args) > 1:
                futures = [
                    self._executor.submit(self._eigh_single_cpu, idx, H_i, force_i)
                    for idx, H_i, force_i in worker_args
                ]
                for future in futures:
                    idx, dpos_i = future.result()
                    dpos_state[self._coord_refs[idx]] = dpos_i
            else:
                for idx, H_i, force_i in worker_args:
                    _, dpos_i = self._eigh_single_cpu(idx, H_i, force_i)
                    dpos_state[self._coord_refs[idx]] = dpos_i

        dpos = dpos_state.to(self.device).reshape(-1, 3)
        steplengths = (dpos ** 2).sum(dim=-1).sqrt()
        self.pos0 = pos_state
        self.forces0 = forces_state
        return dpos, steplengths

    def determine_step(self, dpos: torch.Tensor, steplengths: torch.Tensor) -> torch.Tensor:
        if _has_scatter:
            longest_steps = scatter(
                steplengths, self.optimizable.batch_indices, reduce="max"
            )
        else:
            index = self.optimizable.batch_indices
            src = steplengths
            num_groups = int(index.max().item()) + 1 if index.numel() > 0 else 0
            out = torch.full((num_groups,), float("-inf"), device=src.device, dtype=src.dtype)
            longest_steps = out.scatter_reduce(
                dim=0, index=index, src=src, reduce="amax", include_self=True
            )
        longest_steps = longest_steps[self.optimizable.batch_indices]
        maxstep = longest_steps.new_tensor(self.maxstep)
        scale = longest_steps.reciprocal() * torch.min(longest_steps, maxstep)
        dpos *= scale.unsqueeze(1)
        return dpos

    def update(self, pos: torch.Tensor, forces: torch.Tensor, pos0: torch.Tensor, forces0: torch.Tensor) -> None:
        worker_args = [
            (
                i,
                self._coord_view(pos, i),
                self._coord_view(forces, i),
                self._coord_view(pos0, i),
                self._coord_view(forces0, i),
                bool(self.optimizable.update_mask[i]),
            )
            for i in range(self.batch_size)
        ]

        if self._executor is not None and len(worker_args) > 1:
            futures = [
                self._executor.submit(self._update_single_cpu, *args)
                for args in worker_args
            ]
            for future in futures:
                idx, H_i = future.result()
                self.H[idx] = H_i
        else:
            for args in worker_args:
                idx, H_i = self._update_single_cpu(*args)
                self.H[idx] = H_i
