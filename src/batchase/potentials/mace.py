"""
MACE batch inference potential adapter for batchASE.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence
import numpy as np
import torch

from ..kernels import get_pbc_graph_kernel
from .base import BatchPotential

logger = logging.getLogger("batchase.potentials.mace")


def _atomic_one_hot(atomic_numbers, z_table, device, dtype):
    from mace.tools import atomic_numbers_to_indices, to_one_hot

    indices = atomic_numbers_to_indices(
        atomic_numbers.to("cpu"), z_table=z_table
    )
    one_hot = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    ).to(device)
    return one_hot.to(dtype)


def _swap_edge_rows(edge_index):
    """Swap edge rows so edge_index[0]=sender, edge_index[1]=receiver."""
    return torch.stack([edge_index[1], edge_index[0]])


class _GraphView:
    __slots__ = ("pos", "cell", "natoms", "atomic_numbers", "batch", "ptr")

    def __init__(self, pos, cell, natoms, atomic_numbers, batch, ptr):
        self.pos = pos
        self.cell = cell.view(-1, 3, 3)
        self.natoms = natoms
        self.atomic_numbers = atomic_numbers
        self.batch = batch
        self.ptr = ptr

    def __getitem__(self, key):
        return getattr(self, key)


class MACEBatchBackend:
    """MACE batched potential adapter implementing BatchPotential."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cuda",
        default_dtype: str = "float64",
        enable_cueq: bool = False,
        use_fasteq: bool = False,
        neighbor: str = "auto",
        use_compile: bool = False,
        calculator=None,
        **mace_kwargs,
    ):
        self.kind = "mace"
        self.neighbor = neighbor
        self.default_dtype = default_dtype
        self.dtype = torch.float64 if default_dtype == "float64" else torch.float32
        self.device = torch.device(device)
        self.use_compile = use_compile

        # Process-level FastEq switch in cuequivariance_torch
        try:
            import cuequivariance_torch as cuet
            cuet.set_fasteq_enabled(use_fasteq)
            self.cuet_fasteq_active = cuet.fasteq_enabled()
        except ImportError:
            self.cuet_fasteq_active = False

        if calculator is None:
            from mace.calculators import mace_off
            _mace_dev = str(device).split(":")[0] if str(device).startswith("cuda") else device
            calculator = mace_off(
                model=model,
                device=_mace_dev,
                default_dtype=default_dtype,
                enable_cueq=enable_cueq,
                **mace_kwargs,
            )
        self.calculator = calculator
        self.model = calculator.models[0]
        self.z_table = calculator.z_table

        r_max = getattr(self.model, "r_max", None)
        if r_max is None:
            raise RuntimeError("Model does not contain 'r_max' buffer.")
        self.r_max = float(r_max)

        logger.info(
            "MACEBatchBackend initialized: device=%s dtype=%s r_max=%.2f neighbor=%s enable_cueq=%s use_fasteq=%s",
            self.device,
            self.dtype,
            self.r_max,
            self.neighbor,
            enable_cueq,
            use_fasteq,
        )

    def _run_neighbor_kernel(self, pos, cell, natoms, atomic_numbers, batch, ptr):
        gbatch = _GraphView(
            pos=pos,
            cell=cell,
            natoms=natoms,
            atomic_numbers=atomic_numbers,
            batch=batch,
            ptr=ptr,
        )
        kernel_fn = get_pbc_graph_kernel(self.neighbor)
        edge_index, unit_shifts, num_neighbors = kernel_fn(
            gbatch, radius=self.r_max, pbc=[True, True, True], dtype=self.dtype
        )
        return edge_index, unit_shifts, num_neighbors

    def _build_inputs(self, pos, cell, natoms, atomic_numbers, batch, ptr):
        pos = pos.to(self.device).to(self.dtype).clone()
        cell = cell.to(self.device).to(self.dtype).view(-1, 3).clone()
        natoms = natoms.to(self.device)
        atomic_numbers = atomic_numbers.to(self.device)
        batch = batch.to(self.device)
        ptr = ptr.to(self.device)

        edge_index, unit_shifts, _ = self._run_neighbor_kernel(
            pos, cell, natoms, atomic_numbers, batch, ptr
        )
        edge_index = _swap_edge_rows(edge_index)
        sender = edge_index[0]
        unit_shifts = unit_shifts.to(self.dtype)
        cell_b = cell.view(-1, 3, 3)
        shifts = torch.einsum("be,bec->bc", unit_shifts, cell_b[batch[sender]])

        node_attrs = _atomic_one_hot(
            atomic_numbers, self.z_table, self.device, self.dtype
        )
        return {
            "positions": pos,
            "cell": cell,
            "batch": batch,
            "ptr": ptr,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "node_attrs": node_attrs,
        }

    def build_inputs(self, gbatch) -> dict:
        return self._build_inputs(
            pos=gbatch["pos"],
            cell=gbatch["cell"],
            natoms=gbatch["natoms"],
            atomic_numbers=gbatch["atomic_numbers"],
            batch=gbatch["batch"],
            ptr=gbatch["ptr"],
        )

    def build_inputs_from_atoms(self, atoms_list: Sequence) -> dict:
        natoms = torch.tensor([len(a) for a in atoms_list], dtype=torch.long)
        positions = torch.from_numpy(
            np.concatenate([np.asarray(a.get_positions()) for a in atoms_list])
        )
        cell = torch.from_numpy(
            np.stack([np.asarray(a.get_cell(complete=True)) for a in atoms_list])
        )
        atomic_numbers = torch.from_numpy(
            np.concatenate([np.asarray(a.get_atomic_numbers()) for a in atoms_list])
        )
        batch = torch.repeat_interleave(
            torch.arange(len(atoms_list), dtype=torch.long), natoms
        )
        ptr = torch.cat(
            [torch.tensor([0], dtype=torch.long), natoms.cumsum(0)]
        )
        return self._build_inputs(
            pos=positions,
            cell=cell,
            natoms=natoms,
            atomic_numbers=atomic_numbers,
            batch=batch,
            ptr=ptr,
        )

    def _forward(self, inputs: dict, compute_stress: bool) -> dict:
        out = self.model(
            inputs,
            compute_stress=compute_stress,
            training=self.use_compile,
        )
        results = {
            "energy": out["energy"].unsqueeze(-1).detach().to(torch.float64),
            "forces": out["forces"].detach().to(torch.float64),
        }
        if compute_stress:
            results["stress"] = out["stress"].detach().to(torch.float64)
        return results

    def predict(self, gbatch, compute_stress: bool = False) -> dict:
        return self._forward(self.build_inputs(gbatch), compute_stress)

    def predict_from_atoms(self, atoms_list: Sequence, compute_stress: bool = False) -> dict:
        return self._forward(self.build_inputs_from_atoms(atoms_list), compute_stress)
