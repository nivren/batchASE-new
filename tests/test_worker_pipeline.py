"""
End-to-end pipeline tests for Worker and Scheduler:
1. Slot replenishment and failure isolation when a structure fails (e.g., force_overflow or invalid_cell).
2. Stage 2 isolation: failed structures are strictly barred from entering Stage 2.
3. Single-stage relaxation compatibility (skip_second_stage=True).
4. Summary CSV generation with status and failed_reason fields.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read, write

from batchase.engine.worker import Worker
from batchase.engine.scheduler import Scheduler
from batchase.relaxation import SlotStatus, FailReason

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.expanduser("~/.cache/mace/MACE-OFF23_small.model")
INPUT_CIF = str(HERE / "fixtures/input.cif")


class TestWorkerPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.has_mace = os.path.exists(MODEL_PATH) and torch.cuda.is_available()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stage2_isolation_and_csv_generation(self):
        """Verify failed structures do not enter Stage 2 and CSV correctly captures status/failed_reason."""
        if not self.has_mace:
            self.skipTest("MACE model or CUDA not available for end-to-end Worker test")

        input_dir = os.path.join(self.tmpdir, "inputs")
        output_dir = os.path.join(self.tmpdir, "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        base_atoms = read(INPUT_CIF)

        # 1. First normal structure: will converge
        cif_normal1 = os.path.join(input_dir, "struct_normal1.cif")
        write(cif_normal1, base_atoms)

        # 2. Highly deformed / overlapping atoms structure: force > 100 eV/A -> fails with FORCE_OVERFLOW
        atoms_overflow = base_atoms.copy()
        pos = atoms_overflow.get_positions()
        pos[1] = pos[0] + [0.05, 0.05, 0.05]
        atoms_overflow.set_positions(pos)
        cif_overflow = os.path.join(input_dir, "struct_overflow.cif")
        write(cif_overflow, atoms_overflow)

        # 3. Second normal structure: initially pending in queue, will replenish slot vacated by struct_overflow
        cif_normal2 = os.path.join(input_dir, "struct_normal2.cif")
        write(cif_normal2, base_atoms)

        # Order: [cif_overflow, cif_normal1, cif_normal2] with batch_size=2
        # cif_overflow and cif_normal1 fill initial 2 slots.
        # cif_overflow fails immediately, triggering replenishment with cif_normal2.
        files = [cif_overflow, cif_normal1, cif_normal2]

        worker = Worker(
            files=files,
            device=self.device,
            batch_size=2,
            max_steps=5,
            fmax1=0.05,
            fmax2=0.05,
            f_upper_limit=100.0,
            optimizer1="FIRE",
            optimizer2="FIRE",
            skip_second_stage=False,
            output_path=output_dir,
            model="mace",
            molecule_single=46,
            worker_id=0,
        )

        worker.run()

        # Check Stage 1 outputs
        s1_json_dir = os.path.join(output_dir, "json_result_press")
        self.assertTrue(os.path.exists(s1_json_dir))

        with open(os.path.join(s1_json_dir, "struct_overflow.json"), "r") as f:
            overflow_data = json.load(f)
            self.assertEqual(overflow_data["status"], "failed")
            self.assertEqual(overflow_data["failed_reason"], FailReason.FORCE_OVERFLOW)
            self.assertFalse(overflow_data["converged"])

        with open(os.path.join(s1_json_dir, "struct_normal1.json"), "r") as f:
            normal1_data = json.load(f)
            self.assertEqual(normal1_data["status"], "converged")
            self.assertTrue(normal1_data["converged"])

        # Struct normal 2 must also have run via replenishment
        with open(os.path.join(s1_json_dir, "struct_normal2.json"), "r") as f:
            normal2_data = json.load(f)
            self.assertEqual(normal2_data["status"], "converged")
            self.assertTrue(normal2_data["converged"])

        # Check Stage 2 inputs: strictly only passed structures, failed ones must NOT be in final stage
        s2_json_dir = os.path.join(output_dir, "json_result_final")
        self.assertTrue(os.path.exists(s2_json_dir))
        final_files = os.listdir(s2_json_dir)
        self.assertNotIn("struct_overflow.json", final_files, "Overflow force structure leaked into Stage 2!")
        self.assertIn("struct_normal1.json", final_files)
        self.assertIn("struct_normal2.json", final_files)

        # Verify Scheduler summary CSV generation
        scheduler = Scheduler(
            files=files,
            devices=[self.device],
            batch_size=2,
            output_path=output_dir,
            model="mace",
        )
        scheduler._write_summary_csv()
        csv_file = os.path.join(output_dir, "results_scheduler.csv")
        self.assertTrue(os.path.exists(csv_file))

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {r["file"]: r for r in reader}

        self.assertIn("struct_overflow", rows)
        self.assertEqual(rows["struct_overflow"]["status"], "failed")
        self.assertEqual(rows["struct_overflow"]["failed_reason"], FailReason.FORCE_OVERFLOW)
        self.assertEqual(rows["struct_overflow"]["stage1_status"], "failed")
        self.assertEqual(rows["struct_overflow"]["stage2_status"], "")

        self.assertIn("struct_normal1", rows)
        self.assertEqual(rows["struct_normal1"]["status"], "converged")
        self.assertEqual(rows["struct_normal1"]["stage1_status"], "converged")
        self.assertEqual(rows["struct_normal1"]["stage2_status"], "converged")

        self.assertIn("struct_normal2", rows)
        self.assertEqual(rows["struct_normal2"]["status"], "converged")
        self.assertEqual(rows["struct_normal2"]["stage1_status"], "converged")
        self.assertEqual(rows["struct_normal2"]["stage2_status"], "converged")

    def test_single_stage_pipeline(self):
        """Verify single stage relaxation (skip_second_stage=True) runs properly and reports final status."""
        if not self.has_mace:
            self.skipTest("MACE model or CUDA not available for end-to-end Worker test")

        input_dir = os.path.join(self.tmpdir, "inputs_s1")
        output_dir = os.path.join(self.tmpdir, "outputs_s1")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        base_atoms = read(INPUT_CIF)
        cif_path = os.path.join(input_dir, "struct_1.cif")
        write(cif_path, base_atoms)

        worker = Worker(
            files=[cif_path],
            device=self.device,
            batch_size=1,
            max_steps=3,
            fmax1=0.05,
            skip_second_stage=True,
            output_path=output_dir,
            model="mace",
            molecule_single=46,
            worker_id=0,
        )

        worker.run()

        scheduler = Scheduler(
            files=[cif_path],
            devices=[self.device],
            batch_size=1,
            output_path=output_dir,
            model="mace",
        )
        scheduler._write_summary_csv()
        csv_file = os.path.join(output_dir, "results_scheduler.csv")
        self.assertTrue(os.path.exists(csv_file))

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            row = reader[0]
            self.assertEqual(row["file"], "struct_1")
            self.assertIn(row["status"], ["converged", "failed"])
            self.assertIn("stage1_status", row)
            self.assertEqual(row["stage2_status"], "")


if __name__ == "__main__":
    unittest.main()
