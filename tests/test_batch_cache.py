from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.io import write

from batchase.engine import batching, batch_cache
from batchase.engine.batch_cache import load_or_build_batch_plan


def _concurrent_plan(settings):
    return load_or_build_batch_plan(**settings)[1]["status"]


class TestBatchPlanCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.files = []
        for index, natoms in enumerate((2, 3, 4, 2)):
            path = self.root / f"input_{index}.cif"
            write(path, Atoms("Si" * natoms, positions=[[i, 0, 0] for i in range(natoms)],
                              cell=[10] * 3, pbc=True))
            self.files.append(str(path))

    def plan(self, **kwargs):
        settings = dict(files=self.files, batch_size=2, cache_dir=self.cache)
        settings.update(kwargs)
        return load_or_build_batch_plan(**settings)

    def test_hit_skips_cif_parsing_sorting_and_packing_in_all_modes(self):
        for mode in ("bsize", "atoms"):
            for order in ("rand", "syst", "atom"):
                with self.subTest(mode=mode, order=order):
                    settings = dict(batch_mode=mode, structure_order=order, max_batch_atoms=6)
                    first, miss = self.plan(**settings)
                    self.assertEqual(miss["status"], "miss")
                    with patch.object(batching, "_build_selected_batch_plan", side_effect=AssertionError("replanned")):
                        second, hit = self.plan(files=self.files[::-1], **settings)
                    self.assertEqual(hit["status"], "hit")
                    self.assertEqual(first, second)
                    self.assertEqual(first, batching.build_batch_plan(self.files, batch_size=2, **settings))

    def test_only_effective_parameters_invalidate_grouping(self):
        for mode, changed, inactive in (
                ("bsize", {"batch_size": 3}, {"max_batch_atoms": 999}),
                ("atoms", {"max_batch_atoms": 7}, {"batch_size": 999})):
            with self.subTest(mode=mode):
                common = dict(batch_mode=mode, max_batch_atoms=6, structure_order="rand")
                self.plan(**common)
                self.assertEqual(self.plan(**(common | inactive))[1]["status"], "hit")
                self.assertEqual(self.plan(**(common | changed))[1]["status"], "miss")
                self.assertEqual(self.plan(**(common | {"structure_order_seed": 99}))[1]["status"], "miss")
                self.assertEqual(self.plan(**(common | {"structure_order": "atom"}))[1]["status"], "miss")
        for order in ("atom", "syst"):
            self.plan(structure_order=order)
            plan, report = self.plan(structure_order=order, structure_order_seed=99)
            self.assertEqual(report["status"], "hit")
            self.assertEqual(plan["structure_order_seed"], 99)

    def test_file_size_mtime_and_selected_set_invalidate_cache(self):
        self.plan()
        path = Path(self.files[0])
        with path.open("a") as handle:
            handle.write("\n# changed size\n")
        self.assertEqual(self.plan()[1]["status"], "miss")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.plan()[1]["status"], "miss")
        self.assertEqual(self.plan(files=self.files[:-1])[1]["status"], "miss")
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()

    def test_relative_paths_are_normalized_for_cross_run_reuse(self):
        relative = [os.path.relpath(path) for path in self.files]
        first, _ = self.plan(files=relative)
        second, report = self.plan()
        self.assertEqual(report["status"], "hit")
        self.assertEqual(first, second)
        self.assertTrue(all(os.path.isabs(path) for path in first["ordered_files"]))

    def test_subset_selection_skips_unselected_cifs_and_reuses_cache(self):
        for order in ("syst", "rand", "atom"):
            with self.subTest(order=order):
                selected = batching.select_structure_files(self.files, order, 42, 2)
                with patch.object(batching, "count_structure_atoms", wraps=batching.count_structure_atoms) as count:
                    first, report = self.plan(structure_order=order, num_structures=2)
                self.assertEqual(report["status"], "miss")
                self.assertCountEqual([call.args[0] for call in count.call_args_list], selected)
                self.assertEqual(count.call_count, 2)
                with patch.object(batching, "_build_selected_batch_plan", side_effect=AssertionError("replanned")):
                    second, report = self.plan(structure_order=order, num_structures=2)
                self.assertEqual(report["status"], "hit")
                self.assertEqual(first, second)

    def test_atom_sampling_seed_and_count_are_part_of_cache_identity(self):
        self.plan(structure_order="atom", num_structures=2, structure_order_seed=1)
        self.assertEqual(self.plan(structure_order="atom", num_structures=2,
                                   structure_order_seed=2)[1]["status"], "miss")
        self.assertEqual(self.plan(structure_order="atom", num_structures=3,
                                   structure_order_seed=1)[1]["status"], "miss")
        self.assertEqual(self.plan(structure_order="atom", num_structures=2,
                                   structure_order_seed=1)[1]["status"], "hit")

    def test_unselected_file_changes_do_not_invalidate_a_syst_subset(self):
        _, report = self.plan(structure_order="syst", num_structures=2)
        with open(self.files[-1], "a") as handle:
            handle.write("\n# unselected file changed\n")
        self.assertEqual(self.plan(structure_order="syst", num_structures=2)[1]["status"], "hit")
        self.assertEqual(self.plan(structure_order="syst", num_structures=3)[1]["status"], "miss")

    def test_random_selection_order_is_part_of_cache_identity(self):
        # Same files and seed can yield a different sample order from a different pool.
        selected = self.files[:2]
        with patch.object(batching, "select_structure_files", return_value=selected):
            self.plan(num_structures=2)
        with patch.object(batching, "select_structure_files", return_value=selected[::-1]):
            plan, report = self.plan(num_structures=2)
        self.assertEqual(report["status"], "miss")
        self.assertEqual(plan["ordered_files"], selected[::-1])

    def test_cache_and_parser_version_changes_invalidate_cache(self):
        self.plan()
        for target, attribute, value in ((batch_cache, "CACHE_VERSION", 999),
                                         (batching, "PLANNING_VERSION", 999),
                                         (batch_cache.ase, "__version__", "changed")):
            with self.subTest(attribute=attribute), patch.object(target, attribute, value):
                self.assertEqual(self.plan()[1]["status"], "miss")

    def test_malformed_and_modified_json_is_rebuilt(self):
        first, report = self.plan()
        path = Path(report["path"])
        for replacement in ("{", "null", '{"plan": {}}'):
            with self.subTest(replacement=replacement):
                path.write_text(replacement)
                with self.assertLogs(batch_cache.logger, level="WARNING"):
                    rebuilt, miss = self.plan()
                self.assertEqual(miss["status"], "miss")
                self.assertEqual(rebuilt, first)
        payload = json.loads(path.read_text())
        payload["plan"]["batches"][0]["files"] = []
        path.write_text(json.dumps(payload))
        with self.assertLogs(batch_cache.logger, level="WARNING"):
            rebuilt, miss = self.plan()
        self.assertEqual(rebuilt, first)
        self.assertEqual(miss["status"], "miss")

    def test_unwritable_cache_falls_back_and_atomic_write_cleans_temporary(self):
        blocked = self.root / "blocked"
        blocked.write_text("directory cannot be created here")
        with self.assertLogs(batch_cache.logger, level="WARNING"):
            plan, report = self.plan(cache_dir=blocked)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(plan["num_files"], 4)
        with patch.object(batch_cache.os, "replace", side_effect=PermissionError("denied")), \
                self.assertLogs(batch_cache.logger, level="WARNING"):
            _, report = self.plan()
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(list(self.cache.glob("*.json")), [])
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_multiple_configurations_reuse_and_evict_least_recently_used(self):
        _, oldest = self.plan(structure_order_seed=1, cache_limit=2)
        _, recent = self.plan(structure_order_seed=2, cache_limit=2)
        # Make the timestamps deterministic without relying on sleep.
        os.utime(oldest["path"], ns=(1, 1))
        os.utime(recent["path"], ns=(2, 2))
        _, hit = self.plan(structure_order_seed=1, cache_limit=2)
        self.assertEqual(hit["status"], "hit")
        _, newest = self.plan(structure_order_seed=3, cache_limit=2)
        self.assertTrue(Path(oldest["path"]).exists())
        self.assertFalse(Path(recent["path"]).exists())
        self.assertTrue(Path(newest["path"]).exists())
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)
        self.assertEqual(self.plan(structure_order_seed=2, cache_limit=2)[1]["status"], "miss")

    def test_default_limit_and_lower_limit_on_hit(self):
        for seed in range(21):
            _, latest = self.plan(structure_order_seed=seed)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 20)
        unrelated = self.cache / "notes.json"
        unrelated.write_text("{}")
        _, hit = self.plan(structure_order_seed=20, cache_limit=1)
        self.assertEqual(hit["status"], "hit")
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)
        self.assertTrue(unrelated.exists())
        self.assertTrue(Path(latest["path"]).exists())

    def test_concurrent_writers_publish_valid_plans_and_obey_limit(self):
        settings = [dict(files=self.files, cache_dir=str(self.cache), cache_limit=2,
                         batch_size=2, structure_order_seed=seed) for seed in range(6)]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
            statuses = list(executor.map(_concurrent_plan, settings))
        self.assertEqual(statuses, ["miss"] * 6)
        files = list(self.cache.glob("*.json"))
        self.assertEqual(len(files), 2)
        self.assertEqual(list(self.cache.glob("*.tmp")), [])
        for path in files:
            record = json.loads(path.read_text())
            self.assertEqual(record["plan_sha256"], batch_cache._digest(record["plan"]))

    def test_files_changed_during_parsing_are_not_cached(self):
        original = batching.count_structure_atoms
        def changed(path):
            result = original(path)
            if path == self.files[0]:
                with open(path, "a") as handle:
                    handle.write("\n# modified while planning\n")
            return result
        with patch.object(batching, "count_structure_atoms", side_effect=changed), \
                self.assertLogs(batch_cache.logger, level="WARNING"):
            _, report = self.plan()
        self.assertEqual(report["status"], "inputs_changed")
        self.assertFalse(self.cache.exists())

    def test_disabled_cache_and_invalid_active_parameters(self):
        for directory in (None, "", "none", "NONE"):
            with self.subTest(directory=directory):
                self.assertEqual(self.plan(cache_dir=directory)[1]["status"], "disabled")
        self.assertFalse(self.cache.exists())
        for options in ({"batch_size": 0}, {"batch_mode": "atoms", "max_batch_atoms": 0},
                        {"structure_order": "invalid"}, {"files": self.files * 2},
                        {"cache_limit": 0}, {"cache_limit": -1}, {"num_structures": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)


if __name__ == "__main__":
    unittest.main()
