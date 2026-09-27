"""
Self-contained mock potential backend for testing and CI without MACE or CUDA dependencies.
"""

from __future__ import annotations

import time
from typing import Sequence
import numpy as np
import torch
from ase import Atoms

from .base import BatchPotential


class MockBatchBackend:
    """Mock batched potential adapter implementing BatchPotential.

    Provides an analytical, conservative Lennard-Jones pair potential field
    coupled with an elastic volumetric + shear unit cell restoring term:
    - Atom potential: V_pair(r) = 4*eps * ((sig/r)^12 - (sig/r)^6)
    - Atom force: F_ij = (24*eps/r^2) * (2*(sig/r)^12 - (sig/r)^6) * (r_i - r_j) (conservative)
    - Virial stress: Xi = 0.5 * sum(F_ij (x) r_ij) / V
    - Cell stress: sigma_vol = B * (V - V0) / V0 * I, sigma_dev = G * (C - C0) / a0
    - Fault injection: Overlapping atom pairs (distance < overflow_distance) trigger
      synthetic forces > 100 eV/A (triggering FailReason.FORCE_OVERFLOW).
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        default_dtype: str = "float64",
        sigma: float = 2.0,
        epsilon: float = 0.1,
        B: float = 1.0,
        G: float = 0.5,
        a0: float = 5.43,
        overflow_distance: float = 0.20,
        **kwargs,
    ) -> None:
        self.kind = "mock"
        self.device = torch.device(device)
        self.default_dtype = default_dtype
        self.dtype = torch.float64 if default_dtype == "float64" else torch.float32
        self.r_max: float = 4.5
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.B = float(B)
        self.G = float(G)
        self.a0 = float(a0)
        self.V0 = float(a0 ** 3)
        self.overflow_distance = float(overflow_distance)

        self.mace_time: float = 0.0
        self.graph_time: float = 0.0
        self.forward_calls: int = 0

    def build_inputs(self, gbatch) -> dict:
        """Construct mock inputs from a batched PyG graph."""
        return {
            "pos": gbatch.pos.to(self.device, dtype=self.dtype),
            "cell": gbatch.cell.to(self.device, dtype=self.dtype),
            "batch": gbatch.batch.to(self.device),
            "natoms": getattr(gbatch, "natoms", None),
            "num_graphs": getattr(gbatch, "num_graphs", int(gbatch.batch.max().item() + 1 if len(gbatch.batch) > 0 else 1)),
        }

    def build_inputs_from_atoms(self, atoms_list: Sequence[Atoms]) -> dict:
        """Construct mock inputs directly from a sequence of ASE Atoms objects."""
        natoms_list = [len(a) for a in atoms_list]
        positions = torch.from_numpy(
            np.concatenate([np.asarray(a.get_positions()) for a in atoms_list])
        ).to(self.device, dtype=self.dtype)
        cell = torch.from_numpy(
            np.stack([np.asarray(a.get_cell(complete=True)) for a in atoms_list])
        ).to(self.device, dtype=self.dtype)
        batch = torch.repeat_interleave(
            torch.arange(len(atoms_list), dtype=torch.long, device=self.device),
            torch.tensor(natoms_list, dtype=torch.long, device=self.device),
        )
        return {
            "pos": positions,
            "cell": cell,
            "batch": batch,
            "natoms": torch.tensor(natoms_list, dtype=torch.long, device=self.device),
            "num_graphs": len(atoms_list),
        }

    def _compute_results(self, inputs: dict, compute_stress: bool) -> dict[str, torch.Tensor]:
        pos = inputs["pos"]
        batch_idx = inputs["batch"]
        num_graphs = inputs["num_graphs"]
        cell = inputs["cell"].view(-1, 3, 3)

        forces = torch.zeros_like(pos)
        energies = torch.zeros(num_graphs, dtype=self.dtype, device=self.device)
        virials = torch.zeros((num_graphs, 3, 3), dtype=self.dtype, device=self.device)

        for b in range(num_graphs):
            idx = torch.where(batch_idx == b)[0]
            if len(idx) < 2:
                continue
            p = pos[idx]
            diff = p.unsqueeze(1) - p.unsqueeze(0)  # [n, n, 3], diff[i, j] = p[i] - p[j]
            r = torch.norm(diff, dim=-1)
            r.fill_diagonal_(float("inf"))

            if r.min() < self.overflow_distance:
                # Trigger synthetic force overflow for clashing structures
                forces[idx] += 1e5 * torch.ones_like(forces[idx])
                continue

            s_over_r = self.sigma / r
            sr6 = s_over_r ** 6
            sr12 = sr6 ** 2

            e_pair = 4.0 * self.epsilon * (sr12 - sr6)
            energies[b] = 0.5 * e_pair.sum()

            coef = (24.0 * self.epsilon / (r ** 2)) * (2.0 * sr12 - sr6)
            f_matrix = coef.unsqueeze(-1) * diff
            forces[idx] = f_matrix.sum(dim=1)
            virials[b] = 0.5 * torch.einsum("ija,ijb->ab", f_matrix, diff)

        results = {
            "energy": energies.unsqueeze(-1).to(dtype=torch.float64),
            "forces": forces.to(dtype=torch.float64),
        }

        if compute_stress:
            vol = torch.linalg.det(cell).abs().unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
            # Volumetric restoring stress: sigma_vol = B * (V - V0) / V0 * I
            vol_strain = (vol - self.V0) / self.V0
            eye = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(num_graphs, 1, 1)
            sigma_vol = self.B * vol_strain * eye
            # Deviatoric shape restoring stress towards cubic a0
            target_cell = eye * self.a0
            sigma_dev = self.G * (cell - target_cell) / self.a0
            cell_stress = sigma_vol + sigma_dev
            virial_stress = virials / vol
            stress = cell_stress + virial_stress
            results["stress"] = (0.5 * (stress + stress.transpose(-2, -1))).to(
                dtype=torch.float64
            )

        return results

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        inputs = self.build_inputs(gbatch)
        t1 = time.perf_counter()
        results = self._compute_results(inputs, compute_stress=compute_stress)
        t2 = time.perf_counter()
        self.graph_time += (t1 - t0)
        self.mace_time += (t2 - t1)
        self.forward_calls += 1
        return results

    def predict_from_atoms(
        self, atoms_list: Sequence[Atoms], compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        inputs = self.build_inputs_from_atoms(atoms_list)
        t1 = time.perf_counter()
        results = self._compute_results(inputs, compute_stress=compute_stress)
        t2 = time.perf_counter()
        self.graph_time += (t1 - t0)
        self.mace_time += (t2 - t1)
        self.forward_calls += 1
        return results
