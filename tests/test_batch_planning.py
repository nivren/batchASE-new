from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from batchase.engine.batching import (build_batch_plan, count_structure_atoms,
                                      select_structure_files, order_structure_files)


class TestBatchPlanning(unittest.TestCase):
    def plan(self, counts, **kwargs):
        files = [f"structure_{index:03d}.cif" for index in range(len(counts))]
        sizes = dict(zip(files, counts))
        with patch("batchase.engine.batching.count_structure_atoms", side_effect=sizes.__getitem__):
            return build_batch_plan(files, **kwargs)

    def assert_coverage(self, plan):
        flattened = [path for batch in plan["batches"] for path in batch["files"]]
        self.assertCountEqual(flattened, plan["ordered_files"])
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_bsize_balances_remainder_and_ignores_atom_budget(self):
        plan = self.plan([100] * 103, batch_size=25, max_batch_atoms=1, structure_order="syst")
        self.assertEqual([batch["nfiles"] for batch in plan["batches"]], [21, 21, 21, 20, 20])
        self.assertEqual(plan["max_batch_atoms"], 0)
        self.assert_coverage(plan)

    def test_bsize_boundaries(self):
        for nfiles, limit, expected in [(0, 4, []), (1, 4, [1]), (4, 4, [4]),
                                        (5, 4, [3, 2]), (6, 1, [1] * 6)]:
            with self.subTest(nfiles=nfiles, limit=limit):
                plan = self.plan([1] * nfiles, batch_size=limit)
                self.assertEqual([batch["nfiles"] for batch in plan["batches"]], expected)
                self.assert_coverage(plan)

    def test_atoms_ignores_structure_count_and_backfills(self):
        plan = self.plan([6, 6, 4, 4], batch_mode="atoms", batch_size=1,
                         max_batch_atoms=10, structure_order="syst")
        self.assertEqual([batch["natoms"] for batch in plan["batches"]], [10, 10])
        self.assertEqual([batch["nfiles"] for batch in plan["batches"]], [2, 2])
        self.assertEqual(plan["batches"][0]["files"], ["structure_000.cif", "structure_002.cif"])
        self.assert_coverage(plan)

    def test_atoms_has_no_cardinality_cap(self):
        plan = self.plan([1] * 12, batch_mode="atoms", batch_size=0, max_batch_atoms=10)
        self.assertEqual([batch["nfiles"] for batch in plan["batches"]], [10, 2])
        self.assert_coverage(plan)

    def test_all_orderings_and_modes_preserve_coverage_and_limits(self):
        counts = [1, 7, 3, 9, 2, 5, 6, 4, 8] * 2
        for mode in ("bsize", "atoms"):
            for order in ("rand", "syst", "atom"):
                with self.subTest(mode=mode, order=order):
                    plan = self.plan(counts, batch_mode=mode, batch_size=4,
                                     max_batch_atoms=10, structure_order=order)
                    self.assert_coverage(plan)
                    if mode == "atoms":
                        self.assertTrue(all(batch["natoms"] <= 10 for batch in plan["batches"]))
                    else:
                        sizes = [batch["nfiles"] for batch in plan["batches"]]
                        self.assertLessEqual(max(sizes), 4)
                        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_rand_is_reproducible_and_does_not_touch_global_rng(self):
        random.seed(9)
        original_state = random.getstate()
        first = self.plan([1] * 20, structure_order="rand", structure_order_seed=42)
        self.assertEqual(first, self.plan([1] * 20, structure_order="rand", structure_order_seed=42))
        second = self.plan([1] * 20, structure_order="rand", structure_order_seed=43)
        self.assertNotEqual(first["ordered_files"], second["ordered_files"])
        self.assertEqual(original_state, random.getstate())

    def test_syst_and_atom_ignore_seed(self):
        for order in ("syst", "atom"):
            first = self.plan([3, 1, 3, 2], structure_order=order, structure_order_seed=2)
            second = self.plan([3, 1, 3, 2], structure_order=order, structure_order_seed=3)
            self.assertEqual(first["ordered_files"], second["ordered_files"])
        self.assertEqual(first["ordered_files"], ["structure_000.cif", "structure_002.cif",
                                                  "structure_003.cif", "structure_001.cif"])

    def test_statistics_describe_nonlinear_size_variation(self):
        batch = self.plan([6, 4], batch_mode="atoms", max_batch_atoms=10)["batches"][0]
        self.assertEqual(batch["min_structure_atoms"], 4)
        self.assertEqual(batch["max_structure_atoms"], 6)
        self.assertEqual(batch["sum_squared_atoms"], 52)
        self.assertEqual(batch["atom_budget_fill_ratio"], 1.0)

    def test_rand_and_atom_select_identical_reproducible_subsets(self):
        counts = [index + 1 for index in range(20)]
        files = [f"structure_{index:03d}.cif" for index in range(20)]
        expected = random.Random(42).sample(files, 7)
        for mode in ("bsize", "atoms"):
            with self.subTest(mode=mode):
                settings = dict(num_structures=7, structure_order_seed=42,
                                batch_mode=mode, max_batch_atoms=100)
                rand = self.plan(counts, structure_order="rand", **settings)
                atom = self.plan(counts, structure_order="atom", **settings)
                self.assertEqual(rand["ordered_files"], expected)
                self.assertEqual(atom["ordered_files"], sorted(expected, reverse=True))
                self.assertEqual(rand["num_candidates"], 20)
                self.assertEqual(rand["num_files"], 7)
                self.assert_coverage(rand)
                self.assert_coverage(atom)
                changed = self.plan(counts, structure_order="atom",
                                    **(settings | {"structure_order_seed": 43}))
                self.assertNotEqual(set(atom["ordered_files"]), set(changed["ordered_files"]))

    def test_syst_uses_first_n_and_boundaries_use_all(self):
        first = self.plan([2] * 10, structure_order="syst", num_structures=3)
        self.assertEqual(first["ordered_files"], [f"structure_{i:03d}.cif" for i in range(3)])
        for order in ("syst", "rand", "atom"):
            for count in (0, 10, 100):
                with self.subTest(order=order, count=count):
                    plan = self.plan([2] * 10, structure_order=order, num_structures=count)
                    self.assertEqual(plan["num_files"], 10)
                    self.assert_coverage(plan)

    def test_selection_does_not_parse_cifs_and_atom_reads_only_sample(self):
        files = [f"{i:03d}.cif" for i in range(20)]
        with patch("batchase.engine.batching.count_structure_atoms", side_effect=AssertionError("read during selection")):
            selected = select_structure_files(files[::-1], "atom", 42, 5)
            rand = order_structure_files(files, "rand", 42, num_structures=5)
        self.assertEqual(selected, rand)
        counts = dict(zip(selected, [3, 5, 3, 1, 5]))
        with patch("batchase.engine.batching.count_structure_atoms", side_effect=counts.__getitem__) as reader:
            atom = order_structure_files(files, "atom", 42, num_structures=5)
        self.assertEqual(reader.call_count, 5)
        self.assertEqual(atom, sorted(selected, key=lambda path: (-counts[path], path)))

    def test_atom_does_not_read_oversized_or_unreadable_unselected_files(self):
        files = [f"{i:03d}.cif" for i in range(20)]
        selected = select_structure_files(files, "atom", 42, 5)
        def count(path):
            if path not in selected:
                raise AssertionError("unselected CIF was parsed")
            return 2
        with patch("batchase.engine.batching.count_structure_atoms", side_effect=count) as reader:
            plan = build_batch_plan(files, batch_mode="atoms", max_batch_atoms=10,
                                    structure_order="atom", num_structures=5)
        self.assertEqual(reader.call_count, 5)
        self.assertCountEqual(plan["ordered_files"], selected)

    def test_invalid_active_limits_and_oversized_structure_fail(self):
        for kwargs in ({"batch_size": 0}, {"batch_mode": "atoms", "max_batch_atoms": 0},
                       {"batch_mode": "atoms", "max_batch_atoms": 5},
                       {"batch_mode": "other"}, {"structure_order": "other"},
                       {"num_structures": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan([6], **kwargs)

    def test_duplicate_paths_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_batch_plan(["same.cif", "./same.cif"])

    def test_actual_atom_count_expands_cif_symmetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "symmetry.cif"
            path.write_text("""data_symmetry
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 2
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,-y,-z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.1 0.2 0.3
""", encoding="utf-8")
            self.assertEqual(count_structure_atoms(str(path)), 2)
            with self.assertRaisesRegex(ValueError, "2 > 1"):
                build_batch_plan([str(path)], batch_mode="atoms", max_batch_atoms=1)


if __name__ == "__main__":
    unittest.main()
