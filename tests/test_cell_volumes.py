from __future__ import annotations

import unittest

import torch
from ase import Atoms

from batchase.engine.worker import Worker
from batchase.neighbors import AtomsToGraphs
from batchase.relaxation import (
    FailReason,
    OptimizableBatch,
    OptimizableUnitCellBatch,
)
from batchase.utils import data_list_collater


class EnergyMockBackend:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.kind = "mock"

    def predict(self, batch, compute_stress: bool = False):
        batch_size = batch.num_graphs
        results = {
            "energy": torch.full(
                (batch_size,),
                2.0,
                dtype=self.dtype,
                device=self.device,
            ),
            "forces": torch.zeros(
                batch.pos.shape,
                dtype=self.dtype,
                device=self.device,
            ),
        }
        if compute_stress:
            results["stress"] = torch.zeros(
                (batch_size, 3, 3),
                dtype=self.dtype,
                device=self.device,
            )
        return results


class DensityAtoms:
    def __init__(self, volume: float) -> None:
        self.volume = volume

    def get_volume(self) -> float:
        return self.volume

    def get_masses(self) -> list[float]:
        return [28.085, 28.085]


def create_mock_batch(cells: list[torch.Tensor]):
    converter = AtomsToGraphs(r_edges=False, r_pbc=True)
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
        cell=[2.0, 2.0, 2.0],
        pbc=True,
    )
    batch = data_list_collater(
        [converter.convert(atoms) for _ in cells]
    )
    batch.cell = torch.stack(cells).to(dtype=torch.float64)
    return batch


class TestCellVolumes(unittest.TestCase):
    def test_volumes_are_positive_while_invalid_cells_still_fail(self):
        cells = [
            torch.diag(torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)),
            torch.diag(torch.tensor([2.0, 2.0, -2.0], dtype=torch.float64)),
            torch.diag(torch.tensor([1.0, 1.0, 1e-9], dtype=torch.float64)),
        ]
        optimizable = OptimizableBatch(
            create_mock_batch(cells),
            backend=EnergyMockBackend(),
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            optimizable.get_volumes(),
            torch.tensor([8.0, 8.0, 1e-9], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-15,
        )

        optimizable.converged(
            max_forces=torch.zeros(3, dtype=torch.float64),
            fmax=0.05,
        )
        self.assertEqual(optimizable.converge_indices_list, [0])
        self.assertEqual(optimizable.failed_indices_list, [1, 2])
        self.assertEqual(optimizable.failed_reasons[1], FailReason.INVALID_CELL)
        self.assertEqual(optimizable.failed_reasons[2], FailReason.INVALID_CELL)

    def test_enthalpy_uses_absolute_volume(self):
        inverted_cell = torch.diag(
            torch.tensor([2.0, 2.0, -2.0], dtype=torch.float64)
        )
        optimizable = OptimizableUnitCellBatch(
            create_mock_batch([inverted_cell]),
            backend=EnergyMockBackend(),
            scalar_pressure=0.5,
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            optimizable.get_potential_energies(),
            torch.tensor([6.0], dtype=torch.float64),
        )

    def test_density_rejects_degenerate_and_nonfinite_volumes(self):
        worker = Worker.__new__(Worker)
        expected_density = ((28.085 * 2) / 8.0) * 1.66053906660

        self.assertAlmostEqual(
            worker._get_density(DensityAtoms(8.0)),
            expected_density,
        )
        for volume in (0.0, 1e-9, float("nan"), float("inf")):
            with self.subTest(volume=volume):
                self.assertEqual(worker._get_density(DensityAtoms(volume)), 0.0)


if __name__ == "__main__":
    unittest.main()
