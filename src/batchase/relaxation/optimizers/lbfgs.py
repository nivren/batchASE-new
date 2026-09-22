"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import ase
import torch

try:
    from torch_scatter import scatter
except ImportError:
    scatter = None

if TYPE_CHECKING:
    from ..optimizable import OptimizableBatch


class LBFGS:
    """Limited memory BFGS optimizer for batch ML relaxations."""

    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.02,
        memory: int = 100,
        damping: float = 1.2,
        alpha: float = 100.0,
        save_full_traj: bool = True,
        traj_dir: Path | None = None,
        traj_names: list[str] | None = None,
        early_stop: bool = False,
    ) -> None:
        """
        Args:
            optimizable_batch: an optimizable batch which includes a model and a batch of data
            maxstep: largest step that any atom is allowed to move
            memory: Number of steps to be stored in memory
            damping: The calculated step is multiplied with this number before added to the positions.
            alpha: Initial guess for the Hessian (curvature of energy surface)
            save_full_traj: wether to save full trajectory
            traj_dir: path to save trajectories in
            traj_names: list of trajectory files names
        """
        self.optimizable = optimizable_batch
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.H0 = 1.0 / self.alpha
        self.save_full = save_full_traj
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.trajectories = None

        self.fmax = None
        self.steps = None

        self.s = deque(maxlen=self.memory)
        self.y = deque(maxlen=self.memory)
        self.rho = deque(maxlen=self.memory)
        self.r0 = None
        self.f0 = None

        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"

        self.early_stop  = early_stop

    def run(self, fmax, steps):
        self.fmax = fmax
        self.steps = steps

        self.s.clear()
        self.y.clear()
        self.rho.clear()
        self.r0 = self.f0 = None

        self.trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            self.trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj_tmp", mode="w")
                for name in self.traj_names
            ]

        iteration = 0
        max_forces = self.optimizable.get_max_forces(apply_constraint=True)
        logging.debug("Step   Fmax(eV/A)")
        # print("Step   Fmax(eV/A)")

        while iteration < steps and not self.optimizable.converged(
            forces=None, fmax=self.fmax, max_forces=max_forces
        ):

            if self.early_stop:
                converge_indices_list = self.optimizable.converge_indices_list
                if len(converge_indices_list) > 0:
                    logging.debug(f"Early stopping at iteration {iteration}")
                    break

            logging.debug(
                f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
            )

            # self.energies = self.optimizable.get_potential_energies()
            # logging.info(
            #     f"{iteration} {self.energies.tolist()}"
            # )
            # print(
            #     f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
            # )

            if self.trajectories is not None and (
                self.save_full is True or iteration == 0
            ):
                self.write()

            self.step(iteration)
            max_forces = self.optimizable.get_max_forces(apply_constraint=True)
            iteration += 1

        # logging.info(
        #     f"{iteration} " + " ".join(f"{x:0.3f}" for x in max_forces.tolist())
        # )
        logging.debug(
            f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
        )
        # print(
        #     f"{iteration} " + " ".join(f"{x:18.15g}" for x in max_forces.tolist())
        # )

        # save after converged or all iterations ran
        if iteration > 0 and self.trajectories is not None:
            self.write()

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.close()
            for name in self.traj_names:
                traj_fl = Path(self.traj_dir / f"{name}.traj_tmp", mode="w")
                traj_fl.rename(traj_fl.with_suffix(".traj"))

        # set predicted values to batch
        for name, value in self.optimizable.results.items():
            setattr(self.optimizable.batch, name, value)

        self.nsteps = iteration

        if self.early_stop:
            converge_indices_list = self.optimizable.converge_indices_list
            return converge_indices_list
        else:
            return self.optimizable.converged(
                forces=None, fmax=self.fmax, max_forces=max_forces
            )

    def determine_step(self, dr):
        steplengths = torch.norm(dr, dim=1)
        if device == 'cuda':
            longest_steps = scatter(
                steplengths, self.optimizable.batch_indices, reduce="max"
            )
        else:
            index = self.optimizable.batch_indices
            src = steplengths

            num_groups = int(index.max().item()) + 1

            out = torch.full(
                (num_groups,),
                float('-inf'),
                device=src.device,
                dtype=src.dtype
            )

            longest_steps = out.scatter_reduce(
                dim=0,
                index=index,
                src=src,
                reduce="amax",
                include_self=True
            )
        longest_steps = longest_steps[self.optimizable.batch_indices]
        maxstep = longest_steps.new_tensor(self.maxstep)
        # scale = (longest_steps + 1e-7).reciprocal() * torch.min(longest_steps, maxstep)
        scale = (longest_steps).reciprocal() * torch.min(longest_steps, maxstep)
        dr *= scale.unsqueeze(1)
        return dr * self.damping

    def _batched_dot(self, x: torch.Tensor, y: torch.Tensor):
        if device == 'cuda':
            return scatter(
                (x * y).sum(dim=-1), self.optimizable.batch_indices, reduce="sum"
            )
        else:
            index = self.optimizable.batch_indices
            src = (x * y).sum(dim=-1)   # shape: (N,)
            num_groups = int(index.max().item()) + 1
            out = torch.zeros(
                num_groups, device=src.device, dtype=src.dtype
            )
            out.scatter_add_(dim=0, index=index, src=src)
            return out

    def step(self, iteration: int) -> None:
        # cast forces and positions to float64 otherwise the algorithm is prone to overflow
        forces = self.optimizable.get_forces(apply_constraint=True).to(
            dtype=torch.float64
        )
        pos = self.optimizable.get_positions().to(dtype=torch.float64)

        # Update s, y, rho
        if iteration > 0:
            s0 = pos - self.r0
            self.s.append(s0)
            
            y0 = -(forces - self.f0)
            self.y.append(y0)
            
            self.rho.append(1.0 / self._batched_dot(y0, s0))

            # Debug: Log history buffer state
            # print(f"\nTorch LBFGS History Update (iteration {iteration}):")
            # print("s0", s0.detach().cpu().numpy())
            # print("self.f0", self.f0.detach().cpu().numpy())
            # print("forces", forces.detach().cpu().numpy())
            # print("y0", y0.detach().cpu().numpy())
            # print("rho", self.rho[-1].detach().cpu().numpy())
            # print("Buffer sizes:", len(self.s), "s,", len(self.y), "y,", len(self.rho), "rho")

        loopmax = min(self.memory, iteration)
        alpha = forces.new_empty(loopmax, self.optimizable.batch.natoms.shape[0])
        q = -forces

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = self.rho[i] * self._batched_dot(self.s[i], q)  # b
            q -= alpha[i][self.optimizable.batch_indices, ..., None] * self.y[i]

        z = self.H0 * q
        for i in range(loopmax):
            beta = self.rho[i] * self._batched_dot(self.y[i], z)
            z += self.s[i] * (
                alpha[i][self.optimizable.batch_indices, ..., None]
                - beta[self.optimizable.batch_indices, ..., None]
            )

        # descent direction
        p = -z
        dr = self.determine_step(p)

        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        self.optimizable.set_positions(pos + dr)
        self.r0 = pos
        self.f0 = forces

    def write(self) -> None:
        atoms_objects = self.optimizable.get_atoms_list()
        for atm, traj, mask in zip(
            atoms_objects, self.trajectories, self.optimizable.update_mask
        ):
            if mask:
                traj.write(atm)
