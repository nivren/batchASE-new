from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from ase import Atoms
from ase.io import write

from batchase.engine.worker import Worker
from batchase.neighbors import AtomsToGraphs
from batchase.potentials import create_backend
from batchase.relaxation import OptimizableUnitCellBatch
from batchase.relaxation.optimizers.bfgsfusedls import BFGSFusedLS
from batchase.utils import data_list_collater


def _create_optimizable():
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64)
    batch = data_list_collater([a2g.convert(atoms)])
    backend = create_backend("mock", device="cpu", sigma=2.0, epsilon=0.1, a0=5.43)
    return OptimizableUnitCellBatch(batch=batch, backend=backend, dtype=torch.float64)


class TestBFGSProfiler(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_optimizer(self, use_profiler: bool):
        optimizer = BFGSFusedLS(
            _create_optimizable(),
            device="cpu",
            early_stop=False,
            use_profiler=use_profiler,
            profiler_log_dir=os.path.join(self.tmpdir, "trace"),
            profiler_schedule_config={"wait": 0, "warmup": 0, "active": 1, "repeat": 1},
        )
        optimizer.run(fmax=1e-12, steps=1)
        return optimizer

    def test_disabled_profiler_does_not_collect_timings(self):
        with patch("batchase.relaxation.optimizers.bfgsfusedls.perf_counter") as clock, patch(
            "batchase.relaxation.optimizers.bfgsfusedls.record_function"
        ) as record, patch("batchase.relaxation.optimizers.bfgsfusedls.torch.cuda.synchronize") as sync:
            optimizer = self._run_optimizer(use_profiler=False)

        self.assertEqual(optimizer.get_profiler_breakdown()["measured_total_s"], 0.0)
        self.assertTrue(all(value == 0 for value in optimizer.profiler_calls.values()))
        clock.assert_not_called()
        record.assert_not_called()
        sync.assert_not_called()

    def test_enabled_profiler_collects_all_sections_and_trace_labels(self):
        optimizer = self._run_optimizer(use_profiler=True)
        breakdown = optimizer.get_profiler_breakdown()

        self.assertGreater(breakdown["measured_total_s"], 0.0)
        for section in optimizer._PROFILER_SECTIONS:
            self.assertGreaterEqual(breakdown[f"{section}_s"], 0.0)
            self.assertGreaterEqual(breakdown[f"{section}_ms"], 0.0)
            self.assertGreaterEqual(breakdown[f"{section}_pct"], 0.0)
            self.assertGreater(optimizer.profiler_calls[section], 0)

        percentage_total = sum(breakdown[f"{section}_pct"] for section in optimizer._PROFILER_SECTIONS)
        self.assertAlmostEqual(percentage_total, 100.0, places=6)

        trace_files = []
        for root, _, names in os.walk(os.path.join(self.tmpdir, "trace")):
            trace_files.extend(os.path.join(root, name) for name in names if name.endswith(".json"))
        self.assertTrue(trace_files)
        trace_chunks = []
        for path in trace_files:
            with open(path, encoding="utf-8") as handle:
                trace_chunks.append(handle.read())
        trace_text = "\n".join(trace_chunks)
        for section in optimizer._PROFILER_SECTIONS:
            self.assertIn(f"bfgs::{section}", trace_text)

    def test_reset_profiler_timings_clears_accumulated_values(self):
        optimizer = self._run_optimizer(use_profiler=True)
        optimizer.reset_profiler_timings()
        breakdown = optimizer.get_profiler_breakdown()

        self.assertEqual(breakdown["measured_total_s"], 0.0)
        self.assertTrue(all(value == 0 for value in optimizer.profiler_calls.values()))

    def test_profiler_does_not_change_cpu_trajectory(self):
        disabled = self._run_optimizer(use_profiler=False)
        enabled = self._run_optimizer(use_profiler=True)

        torch.testing.assert_close(
            disabled.optimizable.get_positions(),
            enabled.optimizable.get_positions(),
            rtol=0.0,
            atol=1e-12,
        )

    def test_worker_exports_stage_breakdown(self):
        input_path = os.path.join(self.tmpdir, "struct.cif")
        output_path = os.path.join(self.tmpdir, "output")
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]],
            cell=[5.43, 5.43, 5.43],
            pbc=True,
        )
        write(input_path, atoms)

        Worker(
            files=[input_path],
            device="cpu",
            batch_size=1,
            max_steps=1,
            fmax1=1e-12,
            filter1="UnitCellFilter",
            optimizer1="BFGSFusedLS",
            skip_second_stage=True,
            model="mock",
            output_path=output_path,
            use_profiler=True,
        ).run()

        with open(os.path.join(output_path, "metrics", "worker_0.json"), encoding="utf-8") as handle:
            metrics = json.load(handle)
        breakdown = metrics["stages"]["press"]["profiler_breakdown"]
        self.assertTrue(breakdown["enabled"])
        self.assertGreater(breakdown["measured_total_s"], 0.0)
        self.assertGreaterEqual(breakdown["optimizer_wall_s"], 0.0)
        self.assertGreaterEqual(breakdown["unprofiled_optimizer_s"], 0.0)
        self.assertGreaterEqual(breakdown["unprofiled_optimizer_pct"], 0.0)
        self.assertGreaterEqual(breakdown["profiler_coverage_pct"], 0.0)
        expected_residual = max(
            breakdown["optimizer_wall_s"] - breakdown["measured_total_s"],
            0.0,
        )
        self.assertAlmostEqual(
            breakdown["unprofiled_optimizer_s"], expected_residual, places=12
        )


if __name__ == "__main__":
    unittest.main()
