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

import numpy as np
import torch

from batchase.relaxation import (
    OptimizableBatch,
    OptimizableUnitCellBatch,
    OptimizableFrechetCellBatch,
)
from batchase.engine.scheduler import Scheduler
from batchase.engine.worker import Worker


from ase import Atoms
from ase.io import write
from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater


class MockBackend:
    def __init__(self, energy_per_system: float = -150.0):
        self.kind = "mock"
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.energy_per_system = energy_per_system
        self.mace_time = 0.0
        self.graph_time = 0.0

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

    def test_numpy_energy_components_batch(self):
        """The separated energy API also works with numpy=True and batch size two."""
        cells = [
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
            torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)),
        ]
        gbatch = create_mock_cell_batch(cells)
        obatch = OptimizableUnitCellBatch(
            batch=gbatch,
            backend=MockBackend(energy_per_system=-120.0),
            scalar_pressure=0.0006,
            numpy=True,
            dtype=torch.float64,
        )

        internal = obatch.get_internal_energies()
        pv = obatch.get_pv_terms()
        enthalpy = obatch.get_enthalpies()

        self.assertIsInstance(internal, np.ndarray)
        self.assertIsInstance(pv, np.ndarray)
        self.assertIsInstance(enthalpy, np.ndarray)
        np.testing.assert_allclose(internal, [-120.0, -120.0])
        np.testing.assert_allclose(pv, [0.6, 0.6])
        np.testing.assert_allclose(enthalpy, [-119.4, -119.4])

    def test_worker_json_keeps_legacy_objective_and_exposes_internal_energy(self):
        """Worker JSON keeps the legacy objective while exposing pure internal energy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = os.path.join(tmpdir, "struct.cif")
            write(
                cif_path,
                Atoms(
                    "H2",
                    positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
                    cell=[5.0, 5.0, 5.0],
                    pbc=True,
                ),
            )
            output_dir = os.path.join(tmpdir, "output")
            worker = Worker(
                files=[cif_path],
                device="cpu",
                batch_size=1,
                max_steps=1,
                filter1="UnitCellFilter",
                molecule_single=1,
                output_path=output_dir,
            )
            _, stage_metrics = worker._run_stage(
                files=[cif_path],
                stage_name="press",
                filter_type="UnitCellFilter",
                optimizer_name="FIRE",
                scalar_pressure=0.001,
                backend=MockBackend(energy_per_system=-1.0),
                a2g=AtomsToGraphs(
                    r_edges=False, r_pbc=True, dtype=torch.float64
                ),
                fmax=0.01,
            )

            self.assertEqual(stage_metrics["max_structure_steps"], 1)
            self.assertEqual(stage_metrics["max_structure_file"], "struct")
            self.assertEqual(stage_metrics["max_structure_status"], "failed")

            with open(
                os.path.join(output_dir, "json_result_press", "struct.json"),
                encoding="utf-8",
            ) as f:
                result = json.load(f)

            self.assertAlmostEqual(result["energy_raw_ev"], result["enthalpy_raw_ev"])
            self.assertAlmostEqual(result["internal_energy_raw_ev"], -1.0)
            self.assertAlmostEqual(
                result["enthalpy_raw_ev"],
                result["internal_energy_raw_ev"] + result["pv_raw_ev"],
            )

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
                    "energy_raw_ev": -99.4,
                    "internal_energy_raw_ev": -100.0,
                    "pv_raw_ev": 0.6,
                    "enthalpy_raw_ev": -99.4,
                    "energy_kj_mol": -9648.5,
                    "enthalpy_kj_mol": -9590.609,
                    "energy": -9590.609,
                    "fmax": 0.004,
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
                    "internal_energy_raw_ev": -100.5,
                    "pv_raw_ev": 0.0,
                    "enthalpy_raw_ev": -100.5,
                    "energy_kj_mol": -9696.7425,
                    "enthalpy_kj_mol": -9696.7425,
                    "energy": -9696.7425,
                    "fmax": 0.003,
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
            self.assertIn("stage1_fmax", row)
            self.assertIn("stage2_energy_kj_mol", row)
            self.assertIn("stage2_enthalpy_kj_mol", row)
            self.assertIn("stage2_fmax", row)

            self.assertEqual(float(row["stage1_energy_kj_mol"]), -9648.5)
            self.assertAlmostEqual(float(row["stage1_enthalpy_kj_mol"]), -9590.609)
            self.assertAlmostEqual(float(row["stage2_energy_kj_mol"]), -9696.7425)
            self.assertAlmostEqual(float(row["stage1_fmax"]), 0.004)
            self.assertAlmostEqual(float(row["stage2_fmax"]), 0.003)

    def test_dashboard_single_stage_uses_explicit_step_units(self):
        """Single-stage dashboards omit S2 and expose result-oriented quality metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = os.path.join(tmpdir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, "worker_0.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "worker_id": 0,
                    "device": "cpu",
                    "stages": {
                        "press": {
                            "structures": 2,
                            "steps": 7,
                            "max_structure_steps": 20,
                            "max_structure_file": "formal_c1_2_1460_z1_46",
                            "elapsed_s": 1.0,
                            "mace_s": 0.2,
                            "opt_s": 0.6,
                            "graph_s": 0.1,
                            "io_s": 0.1,
                        }
                    },
                    "total_elapsed_s": 1.0,
                    "total_steps": 7,
                    "peak_vram_gb": 0.0,
                }, f)

            scheduler = Scheduler(
                files=["a.cif", "b.cif"],
                devices=["cpu"],
                output_path=tmpdir,
                skip_second_stage=True,
                scalar_pressure=0.001,
            )
            scheduler.summary_records = [
                {
                    "stage1_status": "converged",
                    "stage1_steps": "10",
                    "stage1_density": "1.0",
                    "stage1_fmax": "0.005",
                    "stage1_enthalpy_kj_mol": "-9.0",
                    "normalization_status": "normalized",
                    "stage2_status": "",
                    "stage2_steps": "0",
                },
                {
                    "stage1_status": "converged",
                    "stage1_steps": "20",
                    "stage1_density": "1.2",
                    "stage1_fmax": "0.002",
                    "stage1_enthalpy_kj_mol": "-8.0",
                    "normalization_status": "normalized",
                    "stage2_status": "",
                    "stage2_steps": "0",
                },
            ]

            with self.assertLogs("batchase.engine.scheduler", level="INFO") as captured:
                scheduler._print_dashboard(1.0)

            output = "\n".join(captured.output)
            self.assertIn("Per-Structure Steps: 30 steps (S1: 30)", output)
            self.assertIn("Avg: 15.0 S1 per attempted struct", output)
            self.assertNotIn("Batch Iterations", output)
            self.assertIn("Convergence Rates", output)
            self.assertIn("S1                   : 2/2 (100.0%)", output)
            self.assertIn("Final fmax [eV/A]", output)
            self.assertIn("below_target=2/2", output)
            self.assertIn("S1 Max (steps/id)", output)
            self.assertIn("20/formal_c1_2_1460_z1_46", output)
            self.assertNotIn("S2", output)
            self.assertIn("Final stage          : Stage 1 only", output)
            self.assertIn("Enthalpy [kJ/mol per molecule]", output)
            self.assertIn("within_5=2/2", output)

    def test_dashboard_stage_averages_use_stage_specific_denominators(self):
        """Stage 2 averages exclude structures that never entered Stage 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = os.path.join(tmpdir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, "worker_0.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "worker_id": 0,
                    "device": "cpu",
                    "stages": {
                        "press": {
                            "structures": 2,
                            "steps": 30,
                            "elapsed_s": 1.0,
                            "max_structure_steps": 20,
                            "max_structure_file": "press_long.cif",
                        },
                        "final": {
                            "structures": 1,
                            "steps": 4,
                            "elapsed_s": 0.2,
                            "max_structure_steps": 4,
                            "max_structure_file": "final_long.cif",
                        },
                    },
                    "total_elapsed_s": 1.2,
                    "total_steps": 34,
                    "peak_vram_gb": 0.0,
                }, f)

            scheduler = Scheduler(files=["a.cif", "b.cif"], devices=["cpu"], output_path=tmpdir)
            scheduler.summary_records = [
                {
                    "stage1_status": "converged",
                    "stage1_steps": "10",
                    "stage1_fmax": "0.02",
                    "stage2_status": "converged",
                    "stage2_steps": "4",
                    "stage2_density": "1.1",
                    "stage2_fmax": "0.004",
                    "stage2_energy_kj_mol": "-10.0",
                    "normalization_status": "normalized",
                },
                {
                    "stage1_status": "converged",
                    "stage1_steps": "20",
                    "stage2_status": "",
                    "stage2_steps": "0",
                    "normalization_status": "normalized",
                },
            ]

            with self.assertLogs("batchase.engine.scheduler", level="INFO") as captured:
                scheduler._print_dashboard(1.0)

            output = "\n".join(captured.output)
            self.assertIn("Avg: 15.0 S1 / 4.0 S2 per attempted struct", output)
            self.assertIn("S1                   : 2/2 (100.0%)", output)
            self.assertIn("S2                   : 1/1 (100.0%)", output)
            self.assertIn("Final                : 1/1 (100.0%)", output)
            self.assertIn("Stage 2 (Final)", output)
            self.assertIn("Final fmax [eV/A]", output)
            self.assertIn("S1 Max (steps/id)", output)
            self.assertIn("20/press_long.cif", output)
            self.assertIn("4/final_long.cif", output)
            self.assertIn("Energy [kJ/mol per molecule]", output)


if __name__ == "__main__":
    unittest.main()
