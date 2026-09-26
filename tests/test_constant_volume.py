from __future__ import annotations

import unittest

import torch
from ase import Atoms

from batchase.neighbors import AtomsToGraphs
from batchase.relaxation import (
    OptimizableFrechetCellBatch,
    OptimizableUnitCellBatch,
)
from batchase.utils import data_list_collater


class StressMockBackend:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.kind = "mock"

    def predict(self, batch, compute_stress: bool = False):
        batch_size = batch.num_graphs
        stress = torch.tensor(
            [
                [1.0, 0.2, 0.1],
                [0.2, 2.0, 0.3],
                [0.1, 0.3, 4.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        slot_scale = torch.arange(
            1,
            batch_size + 1,
            dtype=self.dtype,
            device=self.device,
        )
        return {
            "energy": torch.zeros(
                batch_size,
                dtype=self.dtype,
                device=self.device,
            ),
            "forces": torch.zeros(
                batch.pos.shape,
                dtype=self.dtype,
                device=self.device,
            ),
            "stress": slot_scale.view(-1, 1, 1) * stress,
        }


def create_mock_batch(batch_size: int):
    converter = AtomsToGraphs(r_edges=False, r_pbc=True)
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    return data_list_collater(
        [converter.convert(atoms) for _ in range(batch_size)]
    )


class TestConstantVolume(unittest.TestCase):
    batch_sizes = (1, 2, 3, 4, 25)

    def assert_cell_forces_are_traceless(
        self,
        optimizable: OptimizableUnitCellBatch | OptimizableFrechetCellBatch,
        batch_size: int,
    ) -> None:
        forces = optimizable.get_forces(no_numpy=True)
        cell_forces = forces[optimizable.batch.num_nodes :].view(
            batch_size,
            3,
            3,
        )
        diagonal = cell_forces.diagonal(dim1=-2, dim2=-1)
        trace = diagonal.sum(dim=-1)
        scale = diagonal.abs().sum(dim=-1).clamp_min(1.0)

        self.assertTrue(torch.isfinite(cell_forces).all().item())
        self.assertLess((trace.abs() / scale).max().item(), 1e-14)

        if batch_size > 1:
            slot_scale = torch.arange(
                1,
                batch_size + 1,
                dtype=cell_forces.dtype,
                device=cell_forces.device,
            )
            normalized = cell_forces / slot_scale.view(-1, 1, 1)
            torch.testing.assert_close(
                normalized,
                normalized[0].expand_as(normalized),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_unit_cell_filter_supports_all_batch_sizes(self):
        for batch_size in self.batch_sizes:
            with self.subTest(batch_size=batch_size):
                optimizable = OptimizableUnitCellBatch(
                    create_mock_batch(batch_size),
                    backend=StressMockBackend(),
                    constant_volume=True,
                    dtype=torch.float64,
                )
                self.assert_cell_forces_are_traceless(optimizable, batch_size)

    def test_frechet_cell_filter_supports_all_batch_sizes(self):
        for batch_size in self.batch_sizes:
            with self.subTest(batch_size=batch_size):
                optimizable = OptimizableFrechetCellBatch(
                    create_mock_batch(batch_size),
                    backend=StressMockBackend(),
                    constant_volume=True,
                    dtype=torch.float64,
                )
                self.assert_cell_forces_are_traceless(optimizable, batch_size)


if __name__ == "__main__":
    unittest.main()
