"""
Zero-dependency mock pipeline tests running fully on CPU without MACE or CUDA.
Tests:
1. MockBatchBackend protocol contract and analytical gradient consistency.
2. CPU convergence across all standard optimizers (FIRE, FIRE2, BFGS, LBFGS).
3. End-to-end 2-stage Worker relaxation pipeline on CPU with observable metrics.
4. Slot replenishment, failure isolation (FORCE_OVERFLOW), and Stage 2 barring on CPU.
5. Multi-worker Scheduler execution and summary CSV generation on CPU.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest

import torch
from ase import Atoms
from ase.io import write

from batchase.potentials import BatchPotential, create_backend
from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater
from batchase.relaxation import (
    OptimizableUnitCellBatch,
    FailReason,
)
from batchase.relaxation.optimizers.fire import FIRE, FIRE2
from batchase.relaxation.optimizers.bfgs import BFGS
from batchase.relaxation.optimizers.lbfgs import LBFGS
from batchase.engine.worker import Worker
from batchase.engine.scheduler import Scheduler


def _create_si_atoms(pos_offset: float = 0.0) -> Atoms:
    return Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.35 + pos_offset, 1.35, 1.35]],
        cell=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        pbc=True,
    )


class TestMockPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.tmpdir, "inputs")
        self.output_dir = os.path.join(self.tmpdir, "outputs")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mock_backend_contract_and_derivatives(self):
        """Verify MockBatchBackend satisfies BatchPotential and has conservative gradients."""
        backend = create_backend("mock", device="cpu", sigma=2.0, epsilon=0.1, a0=5.43)
        self.assertIsInstance(backend, BatchPotential)
        self.assertEqual(backend.kind, "mock")
        self.assertEqual(backend.device, torch.device("cpu"))

        atoms1 = _create_si_atoms(pos_offset=0.0)
        atoms2 = _create_si_atoms(pos_offset=0.02)
        a2g = AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64)
        gbatch = data_list_collater([a2g.convert(atoms1), a2g.convert(atoms2)])

        res = backend.predict(gbatch, compute_stress=True)
        self.assertEqual(res["energy"].shape, (2, 1))
        self.assertEqual(res["forces"].shape, (4, 3))
        self.assertEqual(res["stress"].shape, (2, 3, 3))

        # Check Cauchy stress symmetry: sigma = sigma^T
        stress = res["stress"]
        self.assertTrue(torch.allclose(stress, stress.transpose(-2, -1), atol=1e-10))

        # Finite difference gradient check on atom 1 of structure 0
        eps = 1e-5
        pos_orig = gbatch.pos.clone()
        gbatch.pos[1, 0] += eps
        e_plus = backend.predict(gbatch)["energy"][0, 0].item()

        gbatch.pos[1, 0] = pos_orig[1, 0] - eps
        e_minus = backend.predict(gbatch)["energy"][0, 0].item()

        gbatch.pos = pos_orig
        analytical_f = res["forces"][1, 0].item()
        numerical_f = -(e_plus - e_minus) / (2.0 * eps)
        self.assertAlmostEqual(analytical_f, numerical_f, places=3)

        # Test predict_from_atoms
        res_atoms = backend.predict_from_atoms([atoms1, atoms2], compute_stress=True)
        self.assertTrue(torch.allclose(res_atoms["energy"], res["energy"], atol=1e-8))
        self.assertTrue(torch.allclose(res_atoms["forces"], res["forces"], atol=1e-8))

        backend_float32 = create_backend(
            "mock", device="cpu", default_dtype="float32", sigma=2.0, epsilon=0.1
        )
        res_float32 = backend_float32.predict_from_atoms([atoms1], compute_stress=True)
        for value in res_float32.values():
            self.assertEqual(value.dtype, torch.float64)

    def test_mock_cpu_optimizers_convergence(self):
        """Verify all standard optimizers (FIRE, FIRE2, BFGS, LBFGS) converge on CPU with Mock."""
        optimizers = [
            ("FIRE", FIRE, 80),
            ("FIRE2", FIRE2, 80),
            ("BFGS", BFGS, 60),
            ("LBFGS", LBFGS, 60),
        ]

        backend = create_backend("mock", device="cpu", sigma=2.0, epsilon=0.1, a0=5.43)
        a2g = AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64)

        for name, opt_cls, max_steps in optimizers:
            with self.subTest(optimizer=name):
                atoms = _create_si_atoms(pos_offset=0.01)
                gbatch = data_list_collater([a2g.convert(atoms)])
                obatch = OptimizableUnitCellBatch(batch=gbatch, backend=backend, dtype=torch.float64)
                opt = opt_cls(obatch, maxstep=0.2, early_stop=True)
                converged_list = opt.run(fmax=0.01, steps=max_steps)
                self.assertIn(0, converged_list, f"Optimizer {name} failed to converge within {max_steps} steps")
                fmax = obatch.get_max_forces().item()
                self.assertLessEqual(fmax, 0.01, f"Final fmax {fmax} exceeds target 0.01 for {name}")

    def test_mock_worker_two_stage_pipeline_cpu(self):
        """Test complete 2-stage Worker relaxation pipeline on CPU with observable metrics."""
        cif1 = os.path.join(self.input_dir, "struct_1.cif")
        cif2 = os.path.join(self.input_dir, "struct_2.cif")
        write(cif1, _create_si_atoms(pos_offset=0.0))
        write(cif2, _create_si_atoms(pos_offset=0.01))

        worker = Worker(
            files=[cif1, cif2],
            device="cpu",
            batch_size=2,
            max_steps=50,
            fmax1=0.01,
            fmax2=0.01,
            filter1="UnitCellFilter",
            filter2="UnitCellFilter",
            optimizer1="FIRE",
            optimizer2="BFGS",
            model="mock",
            output_path=self.output_dir,
            save_cif=True,
            worker_id=0,
        )
        worker.run()

        # Verify Stage 1 and Stage 2 CIF outputs
        s1_cif = os.path.join(self.output_dir, "cif_result_press", "struct_1.cif")
        s2_cif = os.path.join(self.output_dir, "cif_result_final", "struct_1.cif")
        self.assertTrue(os.path.exists(s1_cif), "Stage 1 CIF should exist")
        self.assertTrue(os.path.exists(s2_cif), "Stage 2 CIF should exist")

        # Verify JSON results and Task 2.8 metrics
        s2_json_path = os.path.join(self.output_dir, "json_result_final", "struct_1.json")
        self.assertTrue(os.path.exists(s2_json_path), "Stage 2 JSON should exist")
        with open(s2_json_path) as f:
            data = json.load(f)

        self.assertEqual(data["status"], "converged")
        self.assertLessEqual(data["fmax"], 0.01)
        self.assertIn("fmax_atom", data)
        self.assertIn("fmax_stress", data)
        self.assertIn("fmax_stress_gpa", data)
        self.assertIsInstance(data["fmax_atom"], float)
        self.assertIsInstance(data["fmax_stress_gpa"], float)
        self.assertIn("energy_raw_ev", data)
        self.assertIn("enthalpy_raw_ev", data)

    def test_mock_worker_replenishment_and_isolation_cpu(self):
        """Test slot replenishment, force overflow detection, and Stage 2 isolation on CPU."""
        # 1. Normal structure 1
        cif_normal1 = os.path.join(self.input_dir, "normal1.cif")
        write(cif_normal1, _create_si_atoms(pos_offset=0.0))

        # 2. Overlapping atoms structure (pos[1] clashing with pos[0] -> distance < 0.1A -> FORCE_OVERFLOW)
        atoms_overflow = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [0.05, 0.05, 0.05]], cell=[5.43, 5.43, 5.43], pbc=True)
        cif_overflow = os.path.join(self.input_dir, "overflow.cif")
        write(cif_overflow, atoms_overflow)

        # 3. Normal structure 2 (queued, will replenish vacated slot)
        cif_normal2 = os.path.join(self.input_dir, "normal2.cif")
        write(cif_normal2, _create_si_atoms(pos_offset=0.01))

        # Initial batch: [cif_overflow, cif_normal1]. cif_overflow fails immediately; cif_normal2 replenishes.
        files = [cif_overflow, cif_normal1, cif_normal2]

        worker = Worker(
            files=files,
            device="cpu",
            batch_size=2,
            max_steps=50,
            fmax1=0.01,
            fmax2=0.01,
            filter1="UnitCellFilter",
            filter2="UnitCellFilter",
            optimizer1="FIRE",
            optimizer2="BFGS",
            model="mock",
            output_path=self.output_dir,
            save_cif=True,
            worker_id=0,
        )
        worker.run()

        # Generate summary CSV
        scheduler = Scheduler(
            files=files,
            devices=["cpu"],
            batch_size=2,
            output_path=self.output_dir,
            model="mock",
        )
        scheduler._write_summary_csv()

        csv_path = os.path.join(self.output_dir, "results_scheduler.csv")
        self.assertTrue(os.path.exists(csv_path), "Summary CSV must exist")

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = {row["file"]: row for row in reader}

        self.assertEqual(len(rows), 3)

        # normal1 and normal2 should converge
        self.assertEqual(rows["normal1"]["status"], "converged")
        self.assertEqual(rows["normal2"]["status"], "converged")

        # overflow structure must be failed with force_overflow
        self.assertEqual(rows["overflow"]["status"], "failed")
        self.assertEqual(rows["overflow"]["failed_reason"], FailReason.FORCE_OVERFLOW)
        self.assertEqual(rows["overflow"]["stage1_status"], "failed")
        self.assertEqual(rows["overflow"]["stage2_status"], "")

        # Confirm failed structure was barred from final output CIF
        final_overflow_cif = os.path.join(self.output_dir, "cif_result_final", "overflow.cif")
        self.assertFalse(os.path.exists(final_overflow_cif), "Failed structure must NOT exist in cif_result_final")

    def test_mock_scheduler_e2e_cpu(self):
        """Test multi-worker Scheduler execution and summary CSV generation on CPU."""
        files = []
        for i in range(4):
            cif_path = os.path.join(self.input_dir, f"struct_{i}.cif")
            write(cif_path, _create_si_atoms(pos_offset=0.01 * i))
            files.append(cif_path)

        scheduler = Scheduler(
            files=files,
            devices=["cpu"],
            num_workers=2,
            batch_size=2,
            max_steps=50,
            fmax1=0.01,
            fmax2=0.01,
            filter1="UnitCellFilter",
            filter2="UnitCellFilter",
            optimizer1="FIRE",
            optimizer2="BFGS",
            model="mock",
            output_path=self.output_dir,
            save_cif=True,
        )
        scheduler.run()

        csv_path = os.path.join(self.output_dir, "results_scheduler.csv")
        self.assertTrue(os.path.exists(csv_path), "results_scheduler.csv should be generated")

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["status"], "converged")
            self.assertIn("stage2_fmax_stress_gpa", row)
