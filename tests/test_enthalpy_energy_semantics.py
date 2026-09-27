"""Tests for distinguishing enthalpy_kj_mol and energy_kj_mol (Task 2.7).

Verifies:
1. OptimizableBatch reports internal energy E, pv_term=0, and enthalpy=E.
2. OptimizableUnitCellBatch and OptimizableFrechetCellBatch report internal energy E,
   pv_term = P * V, and enthalpy = E + P * V.
3. Worker JSON output separates energy_raw_ev, enthalpy_raw_ev, pv_raw_ev,
   energy_kj_mol, and enthalpy_kj_mol.
4. Scheduler summary CSV contains stage1/2 energy and enthalpy columns.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

import torch

from batchase.relaxation import (
    OptimizableBatch,
    OptimizableUnitCellBatch,
    OptimizableFrechetCellBatch,
)
from batchase.engine.scheduler import Scheduler


from ase import Atoms
from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater


class MockBackend:
    def __init__(self, energy_per_system: float = -150.0):
        self.kind = "mock"
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.energy_per_system = energy_per_system

    def predict(self, batch, compute_stress: bool = False):
        batch_size = batch.num_graphs
        energies = torch.full((batch_size,), self.energy_per_system, dtype=torch.float64)
        forces = torch.zeros(batch.pos.shape, dtype=torch.float64)
        res = {"energy": energies, "forces": forces}
        if compute_stress:
            res["stress"] = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
        return res


def create_mock_cell_batch(cells: list[torch.Tensor]):
    converter = AtomsToGraphs(r_edges=False, r_pbc=True)
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    batch = data_list_collater([converter.convert(atoms) for _ in cells])
    batch.cell = torch.stack(cells).to(dtype=torch.float64)
    return batch


class TestEnthalpyEnergySemantics(unittest.TestCase):
    def test_optimizable_batch_zero_pressure(self):
        """OptimizableBatch has no cell relaxation -> PV=0, enthalpy=E."""
        cells = [
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
        ]
        gbatch = create_mock_cell_batch(cells)
        backend = MockBackend(energy_per_system=-50.0)

        obatch = OptimizableBatch(batch=gbatch, backend=backend, dtype=torch.float64)

        e_int = obatch.get_internal_energies()
        pv = obatch.get_pv_terms()
        h = obatch.get_enthalpies()
        e_pot = obatch.get_potential_energies()

        torch.testing.assert_close(e_int, torch.tensor([-50.0, -50.0], dtype=torch.float64))
        torch.testing.assert_close(pv, torch.tensor([0.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(h, e_int)
        torch.testing.assert_close(e_pot, e_int)

    def test_unit_cell_batch_enthalpy_components(self):
        """OptimizableUnitCellBatch separates E, PV, and H = E + PV."""
        cells = [
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
        ]
        gbatch = create_mock_cell_batch(cells)
        backend = MockBackend(energy_per_system=-120.0)
        p = 0.0006  # eV/Å^3 -> PV = 0.0006 * 1000 = 0.6 eV

        obatch = OptimizableUnitCellBatch(
            batch=gbatch, backend=backend, scalar_pressure=p, dtype=torch.float64
        )

        e_int = obatch.get_internal_energies()
        pv = obatch.get_pv_terms()
        h = obatch.get_enthalpies()
        e_obj = obatch.get_potential_energies()

        expected_pv = torch.tensor([0.6, 0.6], dtype=torch.float64)
        expected_h = torch.tensor([-120.0 + 0.6, -120.0 + 0.6], dtype=torch.float64)

        torch.testing.assert_close(e_int, torch.tensor([-120.0, -120.0], dtype=torch.float64))
        torch.testing.assert_close(pv, expected_pv)
        torch.testing.assert_close(h, expected_h)
        torch.testing.assert_close(e_obj, expected_h)  # Objective is enthalpy when pressurized

    def test_frechet_cell_batch_enthalpy_components(self):
        """OptimizableFrechetCellBatch separates E, PV, and H = E + PV."""
        cells = [
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
        ]
        gbatch = create_mock_cell_batch(cells)
        backend = MockBackend(energy_per_system=-80.0)
        p = 0.001  # eV/Å^3 -> PV = 0.001 * 1000 = 1.0 eV

        obatch = OptimizableFrechetCellBatch(
            batch=gbatch, backend=backend, scalar_pressure=p, dtype=torch.float64
        )

        e_int = obatch.get_internal_energies()
        pv = obatch.get_pv_terms()
        h = obatch.get_enthalpies()

        torch.testing.assert_close(e_int, torch.tensor([-80.0, -80.0], dtype=torch.float64))
        torch.testing.assert_close(pv, torch.tensor([1.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close(h, torch.tensor([-79.0, -79.0], dtype=torch.float64))

    def test_scheduler_csv_energy_and_enthalpy_columns(self):
        """Scheduler._write_summary_csv outputs stage1/2 energy and enthalpy columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            s1_dir = os.path.join(tmpdir, "json_result_press")
            s2_dir = os.path.join(tmpdir, "json_result_final")
            os.makedirs(s1_dir, exist_ok=True)
            os.makedirs(s2_dir, exist_ok=True)

            # Stage 1: pressurized (PV work present)
            with open(os.path.join(s1_dir, "struct.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "file": "struct",
                    "status": "converged",
                    "natoms": 46,
                    "num_molecules": 1,
                    "normalization_status": "normalized",
                    "energy_raw_ev": -100.0,
                    "pv_raw_ev": 0.6,
                    "enthalpy_raw_ev": -99.4,
                    "energy_kj_mol": -9648.5,
                    "enthalpy_kj_mol": -9590.609,
                    "energy": -9590.609,
                    "steps": 10,
                    "runtime": 1.0,
                    "density": 1.2,
                }, f)

            # Stage 2: zero pressure
            with open(os.path.join(s2_dir, "struct.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "file": "struct",
                    "status": "converged",
                    "natoms": 46,
                    "num_molecules": 1,
                    "normalization_status": "normalized",
                    "energy_raw_ev": -100.5,
                    "pv_raw_ev": 0.0,
                    "enthalpy_raw_ev": -100.5,
                    "energy_kj_mol": -9696.7425,
                    "enthalpy_kj_mol": -9696.7425,
                    "energy": -9696.7425,
                    "steps": 5,
                    "runtime": 0.5,
                    "density": 1.21,
                }, f)

            scheduler = Scheduler(files=[os.path.join(tmpdir, "struct.cif")], devices=["cpu"], output_path=tmpdir)
            scheduler._write_summary_csv()

            csv_file = os.path.join(tmpdir, "results_scheduler.csv")
            self.assertTrue(os.path.exists(csv_file))

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertIn("stage1_energy_kj_mol", row)
            self.assertIn("stage1_enthalpy_kj_mol", row)
            self.assertIn("stage2_energy_kj_mol", row)
            self.assertIn("stage2_enthalpy_kj_mol", row)

            self.assertEqual(float(row["stage1_energy_kj_mol"]), -9648.5)
            self.assertAlmostEqual(float(row["stage1_enthalpy_kj_mol"]), -9590.609)
            self.assertAlmostEqual(float(row["stage2_energy_kj_mol"]), -9696.7425)


if __name__ == "__main__":
    unittest.main()
