"""
Batched FIRE (Fast Inertial Relaxation Engine) and FIRE2 optimizers for batchASE.

Implements pure GPU tensor-parallel FIRE and FIRE2 algorithms strictly aligned
with ASE's native `ase.optimize.fire.FIRE` and `ase.optimize.fire2.FIRE2`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union
import torch

from ..optimizable import OptimizableBatch
from .base import BatchOptimizer

try:
    from torch_scatter import scatter
    _has_scatter = True
except ImportError:
    scatter = None
    _has_scatter = False

logger = logging.getLogger("batchase.optimizers.fire")


def _batched_dot_per_system(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_indices: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Compute dot product sum_i (x_i * y_i) per system.
    
    Args:
        x, y: [N, 3] tensors
        batch_indices: [N] system assignment for each coordinate row
        num_systems: B total systems
        
    Returns:
        [B] dot product per system
    """
    prod = (x * y).sum(dim=-1)  # [N]
    if _has_scatter and prod.is_cuda:
        return scatter(prod, batch_indices, dim=0, dim_size=num_systems, reduce="sum")
    out = torch.zeros((num_systems,), device=prod.device, dtype=prod.dtype).scatter_add_(
        0, batch_indices, prod
    )
    return out


def _batched_norm_per_system(
    x: torch.Tensor,
    batch_indices: torch.Tensor,
    num_systems: int,
    eps: float = 1e-16,
) -> torch.Tensor:
    """Compute Frobenius / Euclidean norm of vector per system.
    
    Args:
        x: [N, 3] tensor
        batch_indices: [N] system assignment
        num_systems: B total systems
        
    Returns:
        [B] Euclidean norm per system
    """
    sum_sq = _batched_dot_per_system(x, x, batch_indices, num_systems)
    return torch.sqrt(torch.clamp(sum_sq, min=0.0) + eps)


class _BatchFIREBase(BatchOptimizer):
    """
    Unified GPU batched FIRE base optimizer with slot management support.
    """

    def __init__(
        self,
        optimizable: OptimizableBatch,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
        flavor: str = "fire",
        force_reeval: bool = True,
        early_stop: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(optimizable=optimizable, maxstep=maxstep)
        self.dt_start = dt
        self.dtmax = dtmax
        self.dtmin = dtmin
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.flavor = flavor.lower()
        self.force_reeval = force_reeval
        self.early_stop = early_stop

        self.device = torch.device(self.optimizable.device)
        self.dtype = self.optimizable.dtype

        self.initialize()

    def initialize(self) -> None:
        """Initialize optimizer internal states for current batch."""
        B = self.optimizable.batch_size
        pos = self.optimizable.get_positions()
        num_coords = pos.shape[0]

        self.v = torch.zeros((num_coords, 3), device=self.device, dtype=self.dtype)
        self.dt = torch.full((B,), self.dt_start, device=self.device, dtype=self.dtype)
        self.a = torch.full((B,), self.astart, device=self.device, dtype=self.dtype)
        self.Nsteps = torch.zeros((B,), device=self.device, dtype=torch.long)
        self.is_initialized = torch.zeros((B,), device=self.device, dtype=torch.bool)

    def restart_from_earlystop(
        self, restart_indices: List[int], old_batch_indices: torch.Tensor
    ) -> None:
        """Update internal optimizer state when Dynamic Slot Management modifies the batch."""
        new_B = self.optimizable.batch_size
        new_pos = self.optimizable.get_positions()
        new_num_coords = new_pos.shape[0]
        new_batch_indices = self.optimizable.batch_indices

        new_v = torch.zeros((new_num_coords, 3), device=self.device, dtype=self.dtype)
        new_dt = torch.full((new_B,), self.dt_start, device=self.device, dtype=self.dtype)
        new_a = torch.full((new_B,), self.astart, device=self.device, dtype=self.dtype)
        new_Nsteps = torch.zeros((new_B,), device=self.device, dtype=torch.long)
        new_init = torch.zeros((new_B,), device=self.device, dtype=torch.bool)

        for new_idx, old_idx in enumerate(restart_indices):
            if old_idx < len(self.dt):
                new_dt[new_idx] = self.dt[old_idx]
                new_a[new_idx] = self.a[old_idx]
                new_Nsteps[new_idx] = self.Nsteps[old_idx]
                new_init[new_idx] = self.is_initialized[old_idx]

                mask_old = (old_batch_indices == old_idx)
                mask_new = (new_batch_indices == new_idx)
                if mask_old.sum() == mask_new.sum():
                    new_v[mask_new] = self.v[mask_old]

        self.v = new_v
        self.dt = new_dt
        self.a = new_a
        self.Nsteps = new_Nsteps
        self.is_initialized = new_init

    def update_slots(
        self,
        keep_indices: List[int],
        num_new_slots: int,
        old_batch_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Interface alias for slot manager compatibility."""
        if old_batch_indices is None:
            old_batch_indices = self.optimizable.batch_indices
        self.restart_from_earlystop(keep_indices, old_batch_indices)

    def step(self, fmax: float = 0.01) -> None:
        """Perform a single batched FIRE / FIRE2 step."""
        forces = self.optimizable.get_forces(apply_constraint=True)
        if not isinstance(forces, torch.Tensor):
            forces = torch.tensor(forces, device=self.device, dtype=self.dtype)
        else:
            forces = forces.to(device=self.device, dtype=self.dtype)

        pos = self.optimizable.get_positions()
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, device=self.device, dtype=self.dtype)
        else:
            pos = pos.to(device=self.device, dtype=self.dtype)

        B = self.optimizable.batch_size
        batch_idx = self.optimizable.batch_indices.to(device=self.device)
        update_mask = self.optimizable.update_mask.to(device=self.device)

        active_systems = update_mask
        active_atoms = active_systems[batch_idx]

        # Calculate power P = v dot f per system
        vf = _batched_dot_per_system(forces, self.v, batch_idx, B)

        # Check positive/negative power only for systems that completed step 0
        has_history = self.is_initialized & active_systems
        pos_mask = has_history & (vf > 0.0)
        neg_mask = has_history & (vf <= 0.0)

        dt_max_t = torch.tensor(self.dtmax, device=self.device, dtype=self.dtype)
        dt_min_t = torch.tensor(self.dtmin, device=self.device, dtype=self.dtype)

        if self.flavor == "fire":
            # ===== FIRE 1.0 (Strictly matches ase.optimize.fire.FIRE) =====
            # 1. Velocity mixing BEFORE acceleration for positive power systems
            if pos_mask.any():
                pos_atoms = pos_mask[batch_idx]
                f_norm = _batched_norm_per_system(forces, batch_idx, B)
                v_norm = _batched_norm_per_system(self.v, batch_idx, B)

                f_norm_atom = f_norm[batch_idx].unsqueeze(-1)
                v_norm_atom = v_norm[batch_idx].unsqueeze(-1)
                alpha_atom = self.a[batch_idx].unsqueeze(-1)

                safe_mix = pos_atoms & (f_norm_atom.squeeze(-1) > 1e-12)
                v_dir = forces / torch.clamp(f_norm_atom, min=1e-12)
                v_mixed = (1.0 - alpha_atom) * self.v + alpha_atom * v_dir * v_norm_atom
                self.v[safe_mix] = v_mixed[safe_mix]

                inc_mask = pos_mask & (self.Nsteps > self.Nmin)
                self.dt[inc_mask] = torch.minimum(self.dt[inc_mask] * self.finc, dt_max_t)
                self.a[inc_mask] = self.a[inc_mask] * self.fa
                self.Nsteps[pos_mask] += 1

            # 2. Reset on negative power systems
            if neg_mask.any():
                neg_atoms = neg_mask[batch_idx]
                self.v[neg_atoms] = 0.0
                self.a[neg_mask] = self.astart
                self.dt[neg_mask] = torch.maximum(self.dt[neg_mask] * self.fdec, dt_min_t)
                self.Nsteps[neg_mask] = 0

            # 3. Acceleration (Euler)
            dt_atom = self.dt[batch_idx].unsqueeze(-1)
            self.v[active_atoms] += forces[active_atoms] * dt_atom[active_atoms]

        else:
            # ===== FIRE2 (Strictly matches ase.optimize.fire2.FIRE2) =====
            # 1. Adaptation
            if pos_mask.any():
                self.Nsteps[pos_mask] += 1
                inc_mask = pos_mask & (self.Nsteps > self.Nmin)
                self.dt[inc_mask] = torch.minimum(self.dt[inc_mask] * self.finc, dt_max_t)
                self.a[inc_mask] = self.a[inc_mask] * self.fa

            if neg_mask.any():
                self.Nsteps[neg_mask] = 0
                self.dt[neg_mask] = torch.maximum(self.dt[neg_mask] * self.fdec, dt_min_t)
                self.a[neg_mask] = self.astart

                # Backtrack: dr = -0.5 * dt * v
                dt_atom = self.dt[batch_idx].unsqueeze(-1)
                neg_atoms = neg_mask[batch_idx]
                dr_back = -0.5 * dt_atom * self.v
                pos[neg_atoms] += dr_back[neg_atoms]
                self.optimizable.set_positions(pos)
                self.v[neg_atoms] = 0.0

                if self.force_reeval:
                    forces = self.optimizable.get_forces(apply_constraint=True)
                    if not isinstance(forces, torch.Tensor):
                        forces = torch.tensor(forces, device=self.device, dtype=self.dtype)
                    else:
                        forces = forces.to(device=self.device, dtype=self.dtype)

            # 2. Acceleration (Semi-implicit Euler)
            dt_atom = self.dt[batch_idx].unsqueeze(-1)
            self.v[active_atoms] += forces[active_atoms] * dt_atom[active_atoms]

            # 3. Velocity mixing AFTER acceleration
            f_norm = _batched_norm_per_system(forces, batch_idx, B)
            v_norm = _batched_norm_per_system(self.v, batch_idx, B)
            f_norm_atom = f_norm[batch_idx].unsqueeze(-1)
            v_norm_atom = v_norm[batch_idx].unsqueeze(-1)
            alpha_atom = self.a[batch_idx].unsqueeze(-1)

            safe_mix = active_atoms & (f_norm_atom.squeeze(-1) > 1e-12)
            v_dir = forces / torch.clamp(f_norm_atom, min=1e-12)
            v_mixed = (1.0 - alpha_atom) * self.v + alpha_atom * v_dir * v_norm_atom
            self.v[safe_mix] = v_mixed[safe_mix]

        # Calculate step displacement
        dt_atom = self.dt[batch_idx].unsqueeze(-1)
        dr = dt_atom * self.v

        # Zero displacement for inactive systems
        dr[~active_atoms] = 0.0

        # Maximum step clipping per system
        normdr = _batched_norm_per_system(dr, batch_idx, B)  # [B]
        scale = torch.where(
            normdr > self.maxstep,
            self.maxstep / torch.clamp(normdr, min=1e-16),
            torch.ones_like(normdr),
        )
        dr = dr * scale[batch_idx].unsqueeze(-1)

        # Apply displacement
        pos = self.optimizable.get_positions()
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, device=self.device, dtype=self.dtype)
        else:
            pos = pos.to(device=self.device, dtype=self.dtype)

        self.optimizable.set_positions(pos + dr)
        self.is_initialized[active_systems] = True

    def run(
        self,
        fmax: float = 0.01,
        steps: int = 100,
        is_restart_earlystop: bool = False,
        restart_indices: Optional[List[int]] = None,
        old_batch_indices: Optional[torch.Tensor] = None,
    ) -> Union[List[int], bool]:
        """Run optimizer until convergence or steps is reached."""
        self.fmax = fmax
        self.max_iter = steps

        if (
            is_restart_earlystop
            and restart_indices is not None
            and old_batch_indices is not None
        ):
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

            self.step(fmax=self.fmax)
            max_forces = self.optimizable.get_max_forces(apply_constraint=True)
            iteration += 1

        self.nsteps = iteration
        if self.early_stop:
            return self.optimizable.converge_indices_list
        return self.optimizable.converged(forces=None, fmax=self.fmax, max_forces=max_forces)


class FIRE(_BatchFIREBase):
    """
    Batched Fast Inertial Relaxation Engine (FIRE 1.0).
    
    Strictly replicates ASE's default `ase.optimize.fire.FIRE`.
    Uses forward Euler integration with velocity mixing before acceleration.
    """

    def __init__(
        self,
        optimizable: OptimizableBatch,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
        early_stop: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            optimizable=optimizable,
            dt=dt,
            maxstep=maxstep,
            dtmax=dtmax,
            dtmin=dtmin,
            Nmin=Nmin,
            finc=finc,
            fdec=fdec,
            astart=astart,
            fa=fa,
            flavor="fire",
            early_stop=early_stop,
            **kwargs,
        )


class FIRE2(_BatchFIREBase):
    """
    Batched FIRE2 optimizer.
    
    Strictly replicates ASE's `ase.optimize.fire2.FIRE2`.
    Includes backtracking (-0.5 * dt * v), semi-implicit Euler integration,
    and configurable in-step force re-evaluation (`force_reeval`).
    """

    def __init__(
        self,
        optimizable: OptimizableBatch,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 20,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.25,
        fa: float = 0.99,
        force_reeval: bool = True,
        early_stop: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            optimizable=optimizable,
            dt=dt,
            maxstep=maxstep,
            dtmax=dtmax,
            dtmin=dtmin,
            Nmin=Nmin,
            finc=finc,
            fdec=fdec,
            astart=astart,
            fa=fa,
            flavor="fire2",
            force_reeval=force_reeval,
            early_stop=early_stop,
            **kwargs,
        )
