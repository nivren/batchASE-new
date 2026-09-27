from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest

import torch
from ase import Atoms

from batchase.neighbors import AtomsToGraphs
from batchase.relaxation import (
    OptimizableBatch,
    OptimizableFrechetCellBatch,
    OptimizableUnitCellBatch,
)
from batchase.utils import data_list_collater
from batchase.engine.scheduler import Scheduler


class ObservabilityMockBackend:
    def __init__(self, forces: torch.Tensor, stress: torch.Tensor) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.kind = "mock"
        self.forces = forces
        self.stress = stress

    def predict(self, batch, compute_stress: bool = False):
        batch_size = batch.num_graphs
        return {
            "energy": torch.zeros(batch_size, dtype=self.dtype, device=self.device),
            "forces": self.forces.clone(),
            "stress": self.stress.clone(),
        }


def create_two_system_mock_batch():
    converter = AtomsToGraphs(r_edges=False, r_pbc=True)
    atoms1 = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    atoms2 = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    return data_list_collater([converter.convert(atoms1), converter.convert(atoms2)])


class TestResidualObservability(unittest.TestCase):
    def test_optimizable_atom_and_stress_separation(self):
        # System 0: atom 0 force norm = 0.05, atom 1 force norm = 0.01 -> max atom force = 0.05
        # System 1: atom 0 force norm = 0.02, atom 1 force norm = 0.10 -> max atom force = 0.10
        forces = torch.tensor(
            [
                [0.03, 0.04, 0.00],  # sys 0, atom 0 (norm = 0.05)
                [0.01, 0.00, 0.00],  # sys 0, atom 1 (norm = 0.01)
                [0.00, 0.02, 0.00],  # sys 1, atom 0 (norm = 0.02)
                [0.00, 0.06, 0.08],  # sys 1, atom 1 (norm = 0.10)
            ],
            dtype=torch.float64,
        )
        # System 0 stress max component: 0.002
        # System 1 stress max component: 0.005
        stress = torch.zeros((2, 3, 3), dtype=torch.float64)
        stress[0, 0, 1] = 0.002
        stress[0, 1, 0] = 0.002
        stress[1, 2, 2] = -0.005

        batch = create_two_system_mock_batch()
        backend = ObservabilityMockBackend(forces=forces, stress=stress)

        # 1. Test OptimizableBatch (fixed cell)
        opt_batch = OptimizableBatch(
            batch=batch,
            backend=backend,
            compute_stress=False,
            dtype=torch.float64,
        )
        atom_f = opt_batch.get_max_atom_forces()
        self.assertEqual(len(atom_f), 2)
        self.assertAlmostEqual(atom_f[0].item(), 0.05, places=5)
        self.assertAlmostEqual(atom_f[1].item(), 0.10, places=5)

        # Fixed cell residual stress is 0
        stresses = opt_batch.get_max_stresses()
        self.assertAlmostEqual(stresses[0].item(), 0.0)
        self.assertAlmostEqual(stresses[1].item(), 0.0)

        # 2. Test OptimizableUnitCellBatch
        opt_cell = OptimizableUnitCellBatch(
            batch=batch,
            backend=backend,
            scalar_pressure=0.0,
            dtype=torch.float64,
        )
        atom_f_cell = opt_cell.get_max_atom_forces()
        self.assertAlmostEqual(atom_f_cell[0].item(), 0.05, places=5)
        self.assertAlmostEqual(atom_f_cell[1].item(), 0.10, places=5)

        stress_cell = opt_cell.get_max_stresses()
        self.assertAlmostEqual(stress_cell[0].item(), 0.002, places=5)
        self.assertAlmostEqual(stress_cell[1].item(), 0.005, places=5)

        # Combined get_max_forces returns augmented force (not equal to pure stress or atom f alone)
        max_f_aug = opt_cell.get_max_forces()
        self.assertEqual(len(max_f_aug), 2)

    def test_unit_cell_batch_external_pressure_stress(self):
        p_scalar = 0.0006  # eV/Å^3
        batch = create_two_system_mock_batch()

        # Case A: Cauchy stress exactly balances external pressure (sigma = -P * I)
        forces = torch.zeros((4, 3), dtype=torch.float64)
        stress_equil = torch.zeros((2, 3, 3), dtype=torch.float64)
        stress_equil[0] = -p_scalar * torch.eye(3, dtype=torch.float64)
        stress_equil[1] = -p_scalar * torch.eye(3, dtype=torch.float64)

        backend_equil = ObservabilityMockBackend(forces=forces, stress=stress_equil)
        opt_equil = OptimizableUnitCellBatch(
            batch=batch,
            backend=backend_equil,
            scalar_pressure=p_scalar,
            dtype=torch.float64,
        )
        stresses_equil = opt_equil.get_max_stresses()
        # Residual stress should be 0 since sigma + P*I == 0
        self.assertAlmostEqual(stresses_equil[0].item(), 0.0, places=6)
        self.assertAlmostEqual(stresses_equil[1].item(), 0.0, places=6)

        # Case B: Zero Cauchy stress (sigma = 0), residual stress is P * I
        stress_zero = torch.zeros((2, 3, 3), dtype=torch.float64)
        backend_zero = ObservabilityMockBackend(forces=forces, stress=stress_zero)
        opt_zero = OptimizableUnitCellBatch(
            batch=batch,
            backend=backend_zero,
            scalar_pressure=p_scalar,
            dtype=torch.float64,
        )
        stresses_zero = opt_zero.get_max_stresses()
        self.assertAlmostEqual(stresses_zero[0].item(), p_scalar, places=6)
        self.assertAlmostEqual(stresses_zero[1].item(), p_scalar, places=6)

    def test_frechet_cell_batch_residual_stress(self):
        forces = torch.tensor(
            [
                [0.01, 0.00, 0.00],
                [0.00, 0.02, 0.00],
                [0.03, 0.00, 0.00],
                [0.00, 0.04, 0.00],
            ],
            dtype=torch.float64,
        )
        stress = torch.zeros((2, 3, 3), dtype=torch.float64)
        stress[0, 1, 1] = 0.003
        stress[1, 0, 2] = -0.007

        batch = create_two_system_mock_batch()
        backend = ObservabilityMockBackend(forces=forces, stress=stress)

        opt_frechet = OptimizableFrechetCellBatch(
            batch=batch,
            backend=backend,
            scalar_pressure=0.0,
            dtype=torch.float64,
        )
        atom_f = opt_frechet.get_max_atom_forces()
        stresses = opt_frechet.get_max_stresses()

        self.assertAlmostEqual(atom_f[0].item(), 0.02, places=5)
        self.assertAlmostEqual(atom_f[1].item(), 0.04, places=5)
        self.assertAlmostEqual(stresses[0].item(), 0.003, places=5)
        self.assertAlmostEqual(stresses[1].item(), 0.007, places=5)

    def test_worker_and_scheduler_output_residual_metrics(self):
        tmpdir = tempfile.mkdtemp(prefix="test_residual_obs_")
        try:
            s1_dir = os.path.join(tmpdir, "json_result_press")
            s2_dir = os.path.join(tmpdir, "json_result_final")
            os.makedirs(s1_dir, exist_ok=True)
            os.makedirs(s2_dir, exist_ok=True)

            stem = "test_crystal"
            # Write Stage 1 JSON with fmax, fmax_atom, fmax_stress, fmax_stress_gpa
            with open(os.path.join(s1_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "file": stem,
                        "status": "converged",
                        "converged": True,
                        "failed_reason": None,
                        "fmax": 0.008,
                        "fmax_atom": 0.0075,
                        "fmax_stress": 0.0005,
                        "fmax_stress_gpa": 0.0801,
                        "steps": 25,
                        "runtime": 1.2,
                        "natoms": 16,
                        "molecule_single": 8,
                        "num_molecules": 2,
                        "normalization_status": "normalized",
                        "energy_raw_ev": -100.0,
                        "enthalpy_raw_ev": -99.4,
                        "pv_raw_ev": 0.6,
                        "energy_kj_mol": -4824.25,
                        "enthalpy_kj_mol": -4795.31,
                        "energy_per_mol": -4795.31,
                        "energy": -4795.31,
                        "density": 2.1,
                    },
                    f,
                )

            # Write Stage 2 JSON
            with open(os.path.join(s2_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "file": stem,
                        "status": "converged",
                        "converged": True,
                        "failed_reason": None,
                        "fmax": 0.004,
                        "fmax_atom": 0.004,
                        "fmax_stress": 0.0,
                        "fmax_stress_gpa": 0.0,
                        "steps": 10,
                        "runtime": 0.5,
                        "natoms": 16,
                        "molecule_single": 8,
                        "num_molecules": 2,
                        "normalization_status": "normalized",
                        "energy_raw_ev": -100.2,
                        "enthalpy_raw_ev": -100.2,
                        "pv_raw_ev": 0.0,
                        "energy_kj_mol": -4833.9,
                        "enthalpy_kj_mol": -4833.9,
                        "energy_per_mol": -4833.9,
                        "energy": -4833.9,
                        "density": 2.1,
                    },
                    f,
                )

            scheduler = Scheduler(
                files=[os.path.join(tmpdir, f"{stem}.cif")],
                devices=["cpu"],
                output_path=tmpdir,
            )
            scheduler._write_summary_csv()

            csv_file = os.path.join(tmpdir, "results_scheduler.csv")
            self.assertTrue(os.path.exists(csv_file))

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertIn("stage1_fmax", row)
            self.assertIn("stage1_fmax_atom", row)
            self.assertIn("stage1_fmax_stress", row)
            self.assertIn("stage2_fmax", row)
            self.assertIn("stage2_fmax_atom", row)
            self.assertIn("stage2_fmax_stress", row)

            self.assertAlmostEqual(float(row["stage1_fmax"]), 0.008)
            self.assertAlmostEqual(float(row["stage1_fmax_atom"]), 0.0075)
            self.assertAlmostEqual(float(row["stage1_fmax_stress"]), 0.0005)
            self.assertAlmostEqual(float(row["stage2_fmax"]), 0.004)
            self.assertAlmostEqual(float(row["stage2_fmax_atom"]), 0.004)
            self.assertAlmostEqual(float(row["stage2_fmax_stress"]), 0.0)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
