"""
Convert ASE Atoms structures into batched PyTorch Geometric graph representations.
"""

from __future__ import annotations

from typing import Sequence, Optional
import numpy as np
import torch
import ase
from torch_geometric.data import Data


class AtomsToGraphs:
    """
    Converts periodic atomic structures (ASE Atoms) to PyTorch Geometric Data objects.
    
    Supports both on-the-fly graph mode (r_edges=False, where edges are constructed
    by GPU PBC kernels) and static neighbor mode via pymatgen.
    """

    def __init__(
        self,
        max_neigh: int = 200,
        radius: float = 6.0,
        r_energy: bool = False,
        r_forces: bool = False,
        r_distances: bool = False,
        r_edges: bool = False,
        r_fixed: bool = True,
        r_pbc: bool = True,
        r_stress: bool = False,
        r_data_keys: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_stress = r_stress
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges
        self.r_pbc = r_pbc
        self.r_data_keys = r_data_keys
        self.dtype = dtype

    def convert(self, atoms: ase.Atoms, sid=None) -> Data:
        """Convert a single ASE Atoms instance to a PyG Data object."""
        positions = np.array(atoms.get_positions(), copy=True)
        cell = np.array(atoms.get_cell(complete=True), copy=True)

        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.from_numpy(positions).to(dtype=self.dtype)
        cell_tensor = torch.from_numpy(cell).view(1, 3, 3).to(dtype=self.dtype)
        natoms = pos.shape[0]

        tags = torch.tensor(atoms.get_tags(), dtype=torch.long)

        data = Data(
            cell=cell_tensor,
            pos=pos,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
        )

        if sid is not None:
            data.sid = sid

        if self.r_pbc:
            data.pbc = torch.tensor(atoms.pbc, dtype=torch.bool)

        if self.r_fixed:
            fixed_idx = torch.zeros(natoms, dtype=torch.long)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms
                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx

        if self.r_energy:
            try:
                data.energy = atoms.get_potential_energy(apply_constraint=False)
            except Exception:
                pass

        if self.r_forces:
            try:
                data.forces = torch.tensor(
                    atoms.get_forces(apply_constraint=False), dtype=self.dtype
                )
            except Exception:
                pass

        if self.r_stress:
            try:
                data.stress = torch.tensor(
                    atoms.get_stress(apply_constraint=False, voigt=False),
                    dtype=self.dtype,
                )
            except Exception:
                pass

        if self.r_data_keys is not None:
            for data_key in self.r_data_keys:
                if data_key in atoms.info:
                    data[data_key] = atoms.info[data_key]

        return data
