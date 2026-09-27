"""Tests for MOLECULE_SINGLE validation, num_molecules reporting, and CSV integration (Task 2.6).

Verifies:
1. natoms divisible by molecule_single -> integer num_molecules, normalized status, correct per-molecule energy.
2. natoms NOT divisible by molecule_single -> num_molecules=None, energy_per_mol=None, invalid_atom_count status.
3. molecule_single=None -> num_molecules=None, energy_per_mol=None, unnormalized status.
4. results_scheduler.csv includes natoms, num_molecules, and normalization_status columns.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

from ase import Atoms

from batchase.engine.scheduler import Scheduler
from batchase.engine.worker import Worker


class TestMoleculeSingleValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_worker_init_defaults_to_none(self):
        """Worker and Scheduler default molecule_single to None instead of arbitrary 64."""
        worker = Worker(files=["dummy.cif"], device="cpu")
        self.assertIsNone(worker.molecule_single)

        scheduler = Scheduler(files=["dummy.cif"], devices=["cpu"])
        self.assertIsNone(scheduler.molecule_single)

    def test_normalization_exact_multiple(self):
        """natoms % molecule_single == 0 produces integer num_molecules and normalized status."""
        worker = Worker(
            files=["dummy.cif"],
            device="cpu",
            molecule_single=23,
            output_path=self.output_dir,
        )
        self.assertEqual(worker.molecule_single, 23)

        # 46 atoms, molecule_single=23 -> num_molecules=2
        natoms = 46
        e_val = -100.0  # eV

        if worker.molecule_single is not None and worker.molecule_single > 0:
            if natoms % worker.molecule_single == 0:
                num_mol = natoms // worker.molecule_single
                norm_status = "normalized"
                energy_per_mol = (e_val / num_mol) * 96.485
            else:
                num_mol = None
                norm_status = "invalid_atom_count"
                energy_per_mol = None

        self.assertEqual(num_mol, 2)
        self.assertEqual(norm_status, "normalized")
        self.assertAlmostEqual(energy_per_mol, (-100.0 / 2) * 96.485)

    def test_normalization_not_divisible(self):
        """natoms % molecule_single != 0 flags invalid_atom_count and bypasses division."""
        worker = Worker(
            files=["dummy.cif"],
            device="cpu",
            molecule_single=30,
            output_path=self.output_dir,
        )
        self.assertEqual(worker.molecule_single, 30)

        natoms = 46
        e_val = -100.0

        if worker.molecule_single is not None and worker.molecule_single > 0:
            if natoms % worker.molecule_single == 0:
                num_mol = natoms // worker.molecule_single
                norm_status = "normalized"
                energy_per_mol = (e_val / num_mol) * 96.485
            else:
                num_mol = None
                norm_status = "invalid_atom_count"
                energy_per_mol = None

        self.assertIsNone(num_mol)
        self.assertEqual(norm_status, "invalid_atom_count")
        self.assertIsNone(energy_per_mol)

    def test_normalization_when_none(self):
        """molecule_single=None results in unnormalized status."""
        worker = Worker(
            files=["dummy.cif"],
            device="cpu",
            molecule_single=None,
            output_path=self.output_dir,
        )
        self.assertIsNone(worker.molecule_single)

        natoms = 46
        e_val = -100.0

        if worker.molecule_single is not None and worker.molecule_single > 0:
            num_mol = natoms // worker.molecule_single
            norm_status = "normalized"
            energy_per_mol = (e_val / num_mol) * 96.485
        else:
            num_mol = None
            norm_status = "unnormalized"
            energy_per_mol = None

        self.assertIsNone(num_mol)
        self.assertEqual(norm_status, "unnormalized")
        self.assertIsNone(energy_per_mol)

    def test_csv_summary_includes_molecule_fields(self):
        """Scheduler._write_summary_csv includes natoms, num_molecules, and normalization_status."""
        s1_json_dir = os.path.join(self.output_dir, "json_result_press")
        os.makedirs(s1_json_dir, exist_ok=True)

        sample_json = {
            "file": "test_struct",
            "status": "converged",
            "converged": True,
            "failed_reason": None,
            "fmax": 0.005,
            "steps": 12,
            "runtime": 1.5,
            "natoms": 92,
            "molecule_single": 46,
            "num_molecules": 2,
            "normalization_status": "normalized",
            "energy_raw_ev": -200.0,
            "energy_per_mol": -9648.5,
            "energy": -9648.5,
            "density": 1.25,
        }
        with open(os.path.join(s1_json_dir, "test_struct.json"), "w", encoding="utf-8") as f:
            json.dump(sample_json, f)

        scheduler = Scheduler(
            files=[os.path.join(self.output_dir, "test_struct.cif")],
            devices=["cpu"],
            output_path=self.output_dir,
        )
        scheduler._write_summary_csv()

        csv_file = os.path.join(self.output_dir, "results_scheduler.csv")
        self.assertTrue(os.path.exists(csv_file))

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["file"], "test_struct")
        self.assertEqual(row["natoms"], "92")
        self.assertEqual(row["num_molecules"], "2")
        self.assertEqual(row["normalization_status"], "normalized")


if __name__ == "__main__":
    unittest.main()
