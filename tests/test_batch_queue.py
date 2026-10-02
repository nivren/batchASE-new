from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ase import Atoms
from ase.io import write

from batchase.engine.scheduler import Scheduler
from batchase.engine.worker import Worker


class TestBatchQueue(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = []
        for index in range(7):
            path = self.root / f"input_{index}.cif"
            atoms = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
                          cell=[5.43] * 3, pbc=True)
            write(path, atoms)
            self.files.append(str(path))

    def scheduler(self, **kwargs):
        settings = dict(files=self.files, devices=["cpu"], num_workers=2,
                        batch_size=2, max_steps=50, fmax=0.01,
                        optimizer1="FIRE", optimizer2="BFGS",
                        filter1="UnitCellFilter", filter2="UnitCellFilter",
                        model="mock", output_path=str(self.root / "output"))
        settings.update(kwargs)
        return Scheduler(**settings)

    def read_json(self, relative):
        return json.loads((self.root / "output" / relative).read_text())

    def test_shared_queue_processes_every_batch_once_and_keeps_two_stages(self):
        self.scheduler().run()
        plan = self.read_json("batch_plan.json")
        self.assertEqual([batch["nfiles"] for batch in plan["batches"]], [2, 2, 2, 1])
        worker_records = [self.read_json(f"metrics/worker_{index}.json") for index in range(2)]
        completed = [batch_id for record in worker_records for batch_id in record["batch_ids"]]
        self.assertCountEqual(completed, range(4))
        self.assertEqual(len(completed), len(set(completed)))
        self.assertTrue(any(record["num_batches"] > 1 for record in worker_records))
        for batch in plan["batches"]:
            record = self.read_json(f"metrics/batch_{batch['batch_id']}.json")
            self.assertEqual(record["files"], batch["files"])
            self.assertEqual(record["stages"]["press"]["structures"], batch["nfiles"])
            self.assertEqual(record["stages"]["final"]["structures"], batch["nfiles"])
        for worker in worker_records:
            for stage in ("press", "final"):
                total = sum(batch["stages"][stage]["structures"] for batch in worker["batches"])
                self.assertEqual(worker["stages"][stage]["structures"], total)

    def test_atoms_mode_ignores_bsize_and_limits_both_stages(self):
        self.scheduler(batch_mode="atoms", batch_size=1, max_batch_atoms=6).run()
        plan = self.read_json("batch_plan.json")
        self.assertEqual([batch["nfiles"] for batch in plan["batches"]], [3, 3, 1])
        for batch in plan["batches"]:
            record = self.read_json(f"metrics/batch_{batch['batch_id']}.json")
            for stage in record["stages"].values():
                self.assertLessEqual(stage["max_observed_batch_atoms"], 6)
                self.assertEqual(stage["max_batch_atoms"], 6)

    def test_cached_plan_keeps_new_run_ids_and_can_change_worker_configuration(self):
        cache_dir = self.root / "cache"
        first = self.scheduler(num_structures=2, num_workers=1, batch_size=1,
                               batch_plan_cache_dir=str(cache_dir), skip_second_stage=True)
        first.run()
        initial = self.read_json("batch_plan.json")
        self.assertEqual(initial["planning_cache"]["status"], "miss")
        other_output = self.root / "second_output"
        with patch("batchase.engine.batching._build_selected_batch_plan", side_effect=AssertionError("replanned")):
            first.run()
        self.assertEqual(self.read_json("batch_plan.json")["planning_cache"]["status"], "hit")
        second = self.scheduler(num_structures=2, num_workers=2, batch_size=1,
                                batch_plan_cache_dir=str(cache_dir), optimizer1="BFGS",
                                skip_second_stage=True, output_path=str(other_output))
        with patch("batchase.engine.batching._build_selected_batch_plan", side_effect=AssertionError("replanned")):
            second.run()
        reused = json.loads((other_output / "batch_plan.json").read_text())
        self.assertEqual(reused["planning_cache"]["status"], "hit")
        self.assertEqual(reused["active_workers"], 2)
        self.assertNotEqual(initial["run_id"], reused["run_id"])
        self.assertEqual(initial["batches"], reused["batches"])
        self.assertEqual(reused["num_candidates"], 7)
        self.assertEqual(reused["num_files"], 2)
        payload = json.loads(Path(initial["planning_cache"]["path"]).read_text())["plan"]
        self.assertNotIn("run_id", payload)
        self.assertNotIn("num_workers", payload)
        self.assertNotIn("planning_cache", payload)
        for batch in reused["batches"]:
            record = json.loads((other_output / f"metrics/batch_{batch['batch_id']}.json").read_text())
            self.assertEqual(record["run_id"], reused["run_id"])

    def test_fewer_batches_than_workers_and_single_stage(self):
        self.scheduler(files=self.files[:1], num_workers=3, skip_second_stage=True).run()
        plan = self.read_json("batch_plan.json")
        self.assertEqual(plan["active_workers"], 1)
        self.assertFalse((self.root / "output/metrics/worker_1.json").exists())
        self.assertNotIn("final", self.read_json("metrics/worker_0.json")["stages"])

    def test_fixed_external_plan_preserves_ownership_order_and_oversized_cardinality(self):
        assignments = [self.files[:4][::-1], self.files[4:]]
        path = self.root / "fixed.json"
        path.write_text(json.dumps({"workers": [{"files": files} for files in assignments]}))
        cache_dir = self.root / "unused_cache"
        self.scheduler(fixed_batch_plan=str(path), batch_size=1, max_batch_atoms=8,
                       batch_plan_cache_dir=str(cache_dir),
                       batch_mode="atoms", structure_order="rand").run()
        self.assertFalse(cache_dir.exists())
        for worker_id, files in enumerate(assignments):
            record = self.read_json(f"metrics/worker_{worker_id}.json")
            self.assertEqual(record["batch_ids"], [worker_id])
            self.assertEqual(record["batches"][0]["files"], files)
            self.assertEqual(record["stages"]["press"]["structures"], len(files))

    def test_invalid_fixed_plans_fail_before_workers_start(self):
        for workers in ([self.files], [self.files[:1], self.files[:1]],
                        [self.files[:1], self.files[2:]]):
            with self.subTest(workers=workers):
                path = self.root / "fixed.json"
                path.write_text(json.dumps(workers))
                scheduler = self.scheduler(fixed_batch_plan=str(path))
                with patch.object(scheduler, "_execute_workers") as launch:
                    with self.assertRaises(ValueError):
                        scheduler.run()
                    launch.assert_not_called()

    def test_worker_failure_is_reported_by_scheduler(self):
        with self.assertRaisesRegex(RuntimeError, "exit code"):
            self.scheduler(files=self.files[:2], model="unknown").run()
        self.assertFalse((self.root / "output/summary_scheduler.csv").exists())

    def test_no_completion_records_is_not_success(self):
        scheduler = self.scheduler()
        with patch.object(scheduler, "_execute_workers"):
            with self.assertRaisesRegex(RuntimeError, "Missing completion"):
                scheduler.run()

    def test_stale_completion_records_do_not_satisfy_a_new_run(self):
        scheduler = self.scheduler(files=self.files[:1], num_workers=1)
        scheduler.run()
        with patch.object(scheduler, "_execute_workers"):
            with self.assertRaisesRegex(RuntimeError, "Invalid completion"):
                scheduler.run()

    def test_backend_is_loaded_once_and_optimizer_is_new_for_each_batch(self):
        from batchase.engine import worker as module
        from batchase.potentials import create_backend
        from batchase.relaxation import get_optimizer_cls
        backend = create_backend("mock", device="cpu")
        constructed = []

        def factory(name):
            cls = get_optimizer_cls(name)

            def construct(*args, **kwargs):
                optimizer = cls(*args, **kwargs)
                constructed.append(optimizer)
                return optimizer
            return construct

        worker = Worker(self.files[:3], device="cpu", batch_size=2,
                        optimizer1="FIRE", optimizer2="BFGS", max_steps=50,
                        filter1="UnitCellFilter", filter2="UnitCellFilter", model="mock",
                        output_path=str(self.root / "output"))
        with patch.object(module, "create_backend", return_value=backend) as loader, \
                patch.object(module, "get_optimizer_cls", side_effect=factory):
            worker.run()
        loader.assert_called_once()
        self.assertEqual(len(constructed), 4)
        self.assertEqual(len({id(optimizer) for optimizer in constructed}), 4)
        self.assertEqual(self.read_json("metrics/worker_0.json")["num_batches"], 2)

    def _check_all_optimizers(self, device, cpu_threads):
        mixed = self.root / "input_1.cif"
        write(mixed, Atoms("Si3", positions=[[0, 0, 0], [1.36, 1.36, 1.36], [4, 4, 4]],
                          cell=[5.43] * 3, pbc=True))
        for name in ("BFGS", "QuasiNewton", "BFGSFusedLS", "BFGSLineSearch", "LBFGS", "FIRE", "FIRE2"):
            with self.subTest(optimizer=name, device=device):
                output = self.root / name
                Worker(self.files[:3], device=device, batch_mode="atoms", batch_size=1,
                       max_batch_atoms=6, structure_order="syst", optimizer1=name,
                       optimizer2=name, bfgs_cpu_thread=cpu_threads, max_steps=3,
                       fmax1=1e-12, fmax2=1e-12, stage2_include_unconverged=True,
                       filter1="UnitCellFilter", filter2="UnitCellFilter", model="mock",
                       output_path=str(output)).run()
                metrics = json.loads((output / "metrics/worker_0.json").read_text())
                self.assertEqual(metrics["num_batches"], 2)
                self.assertEqual([batch["natoms"] for batch in metrics["batches"]], [5, 2])
                for stage in ("press", "final"):
                    self.assertEqual(metrics["stages"][stage]["structures"], 3)
                    self.assertGreater(metrics["stages"][stage]["steps"], 0)
                    self.assertLessEqual(metrics["stages"][stage]["max_observed_batch_atoms"], 6)

    def test_all_optimizers_process_successive_ragged_batches_cpu(self):
        self._check_all_optimizers("cpu", 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_all_optimizers_process_successive_ragged_batches_gpu(self):
        self._check_all_optimizers("cuda:0", 0)


if __name__ == "__main__":
    unittest.main()
