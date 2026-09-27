from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from batchase.engine.scheduler import Scheduler
from batchase.engine.worker import Worker
from batchase.neighbors import AtomsToGraphs
from batchase.relaxation import (
    OptimizableBatch,
    OptimizableFrechetCellBatch,
    OptimizableUnitCellBatch,
)
from batchase.utils import data_list_collater


class ObservabilityBackend:
    def __init__(self, forces: torch.Tensor | None = None, stress: torch.Tensor | None = None):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.kind = "mock"
        self.forces = forces
        self.stress = stress
        self.mace_time = 0.0
        self.graph_time = 0.0
        self.predict_calls = 0

    def predict(self, batch, compute_stress: bool = False):
        self.predict_calls += 1
        batch_size = batch.num_graphs
        forces = self.forces
        if forces is None:
            forces = torch.zeros_like(batch.pos, dtype=self.dtype)
        stress = self.stress
        if stress is None:
            stress = torch.zeros((batch_size, 3, 3), dtype=self.dtype)
        return {
            "energy": torch.zeros(batch_size, dtype=self.dtype),
            "forces": forces.clone(),
            "stress": stress.clone(),
        }


def create_batch(batch_size: int = 1):
    converter = AtomsToGraphs(r_edges=False, r_pbc=True)
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    return data_list_collater([converter.convert(atoms) for _ in range(batch_size)])


class TestResidualObservability(unittest.TestCase):
    def test_atom_force_and_stress_separation(self):
        forces = torch.tensor(
            [
                [0.03, 0.04, 0.00],
                [0.01, 0.00, 0.00],
                [0.00, 0.02, 0.00],
                [0.00, 0.06, 0.08],
            ],
            dtype=torch.float64,
        )
        stress = torch.zeros((2, 3, 3), dtype=torch.float64)
        stress[0, 0, 1] = 0.002
        stress[0, 1, 0] = 0.002
        stress[1, 2, 2] = -0.005

        opt = OptimizableUnitCellBatch(
            create_batch(2),
            backend=ObservabilityBackend(forces, stress),
            dtype=torch.float64,
        )
        final_forces = opt.get_forces(no_numpy=True)
        atom_fmax = opt.get_max_atom_forces(final_forces)
        stress_fmax = opt.get_max_stresses()

        torch.testing.assert_close(
            atom_fmax,
            torch.tensor([0.05, 0.10], dtype=torch.float64),
        )
        torch.testing.assert_close(
            stress_fmax,
            torch.tensor([0.002, 0.005], dtype=torch.float64),
        )
        self.assertEqual(opt.get_max_forces(final_forces).shape[0], 2)

        fixed = OptimizableBatch(
            create_batch(2),
            backend=ObservabilityBackend(forces, stress),
            compute_stress=False,
            dtype=torch.float64,
        )
        self.assertIsNone(fixed.get_max_stresses())

        measured = OptimizableBatch(
            create_batch(2),
            backend=ObservabilityBackend(forces, stress),
            compute_stress=True,
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            measured.get_max_stresses(),
            torch.tensor([0.002, 0.005], dtype=torch.float64),
        )

    def test_pressure_and_constraint_projection(self):
        pressure = 0.0006
        equilibrium_stress = -pressure * torch.eye(3, dtype=torch.float64).unsqueeze(0)
        equilibrium = OptimizableUnitCellBatch(
            create_batch(),
            backend=ObservabilityBackend(stress=equilibrium_stress),
            scalar_pressure=pressure,
            dtype=torch.float64,
        )
        self.assertLess(equilibrium.get_max_stresses()[0].item(), 1e-9)

        zero_stress = OptimizableUnitCellBatch(
            create_batch(),
            backend=ObservabilityBackend(
                stress=torch.zeros((1, 3, 3), dtype=torch.float64)
            ),
            scalar_pressure=pressure,
            dtype=torch.float64,
        )
        self.assertAlmostEqual(zero_stress.get_max_stresses()[0].item(), pressure, places=9)

        anisotropic = torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 7.0]]],
            dtype=torch.float64,
        )
        constant_volume = OptimizableUnitCellBatch(
            create_batch(),
            backend=ObservabilityBackend(stress=anisotropic),
            constant_volume=True,
            dtype=torch.float64,
        )
        constant_volume.get_forces(no_numpy=True)
        stress_tensor = constant_volume.stress.reshape(1, 3, 3)
        self.assertLess(abs(torch.trace(stress_tensor[0]).item()), 1e-12)

        masked = OptimizableUnitCellBatch(
            create_batch(),
            backend=ObservabilityBackend(stress=anisotropic),
            mask=torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            dtype=torch.float64,
        )
        masked.get_forces(no_numpy=True)
        masked_stress = masked.stress.reshape(1, 3, 3)[0]
        self.assertAlmostEqual(masked_stress[0, 1].item(), 0.0, places=12)
        self.assertAlmostEqual(masked_stress[1, 0].item(), 0.0, places=12)

        frechet = OptimizableFrechetCellBatch(
            create_batch(),
            backend=ObservabilityBackend(stress=anisotropic),
            constant_volume=True,
            dtype=torch.float64,
        )
        frechet.get_forces(no_numpy=True)
        frechet_stress = frechet.stress.reshape(1, 3, 3)
        self.assertLess(abs(torch.trace(frechet_stress[0]).item()), 1e-12)

    def test_nonidentity_deformation_uses_effective_stress(self):
        raw_stress = torch.diag(torch.tensor([10.0, 2.0, 1.0], dtype=torch.float64)).unsqueeze(0)
        for cls in (OptimizableUnitCellBatch, OptimizableFrechetCellBatch):
            with self.subTest(filter=cls.__name__):
                opt = cls(
                    create_batch(),
                    backend=ObservabilityBackend(stress=raw_stress),
                    dtype=torch.float64,
                )
                opt.batch.cell[0] = torch.diag(
                    torch.tensor([6.0, 5.0, 5.0], dtype=torch.float64)
                )
                opt.get_forces(no_numpy=True)
                expected = opt.stress.reshape(1, 3, 3).abs().amax(dim=(1, 2))
                torch.testing.assert_close(opt.get_max_stresses(), expected)

    def test_numpy_and_constraint_contract(self):
        batch = create_batch()
        batch.fixed = torch.tensor([1, 0], dtype=torch.long)
        forces = torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=torch.float64)
        opt = OptimizableUnitCellBatch(
            batch,
            backend=ObservabilityBackend(forces=forces),
            numpy=True,
            dtype=torch.float64,
        )
        augmented = opt.get_forces()
        atom_fmax = opt.get_max_atom_forces(augmented, apply_constraint=True)
        stress_fmax = opt.get_max_stresses()

        self.assertIsInstance(atom_fmax, np.ndarray)
        self.assertIsInstance(stress_fmax, np.ndarray)
        self.assertAlmostEqual(float(atom_fmax[0]), 0.4, places=12)

    def test_worker_json_and_scheduler_csv(self):
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
            worker = Worker(
                files=[cif_path],
                device="cpu",
                batch_size=1,
                max_steps=1,
                filter1="UnitCellFilter",
                molecule_single=1,
                output_path=tmpdir,
            )
            worker._run_stage(
                files=[cif_path],
                stage_name="press",
                filter_type="UnitCellFilter",
                optimizer_name="FIRE",
                scalar_pressure=0.001,
                backend=ObservabilityBackend(),
                a2g=AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64),
                fmax=0.01,
            )

            with open(
                os.path.join(tmpdir, "json_result_press", "struct.json"),
                encoding="utf-8",
            ) as f:
                result = json.load(f)
            self.assertIn("fmax_atom", result)
            self.assertIn("fmax_stress", result)
            self.assertIn("fmax_stress_gpa", result)
            self.assertIsNotNone(result["fmax_stress"])
            self.assertAlmostEqual(
                result["fmax_stress_gpa"],
                result["fmax_stress"] * 160.21766208,
                places=8,
            )

            scheduler = Scheduler(
                files=[cif_path], devices=["cpu"], output_path=tmpdir
            )
            scheduler._write_summary_csv()
            with open(os.path.join(tmpdir, "results_scheduler.csv"), encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertIn("stage1_fmax_atom", row)
            self.assertIn("stage1_fmax_stress", row)
            self.assertIn("stage1_fmax_stress_gpa", row)
            self.assertEqual(row["stage2_fmax_atom"], "")
            self.assertAlmostEqual(
                float(row["stage1_fmax_stress_gpa"]),
                float(row["stage1_fmax_stress"]) * 160.21766208,
                places=8,
            )

            scheduler.summary_records[0]["stage1_status"] = "converged"
            summary = "\n".join(
                scheduler._build_final_structure_summary(
                    scheduler.summary_records, has_stage2=False
                )
            )
            self.assertIn("Final atom fmax [eV/A]", summary)
            self.assertIn("Final stress [GPa]", summary)


if __name__ == "__main__":
    unittest.main()
