from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/batch_relax.py"
spec = importlib.util.spec_from_file_location("batch_relax_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class TestBatchCLI(unittest.TestCase):
    def test_new_options_are_available(self):
        argv = [str(SCRIPT), "--target_folder", "/tmp/inputs", "--batch_mode", "atoms",
                "--max_batch_atoms", "100", "--structure_order", "rand",
                "--structure_order_seed", "17", "--batch_plan_cache_dir", "/tmp/plans",
                "--batch_plan_cache_limit", "7"]
        with patch.object(sys, "argv", argv):
            args = cli.parse_args()
        self.assertEqual((args.batch_mode, args.max_batch_atoms, args.structure_order,
                          args.structure_order_seed), ("atoms", 100, "rand", 17))
        self.assertEqual(args.batch_plan_cache_dir, "/tmp/plans")
        self.assertEqual(args.batch_plan_cache_limit, 7)

    def test_cli_cache_default_and_disable_values(self):
        for extra, expected in (([], ".cache/batch_plans"),
                                (["--batch_plan_cache_dir", ""], ""),
                                (["--batch_plan_cache_dir", "none"], "none")):
            with self.subTest(extra=extra), patch.object(sys, "argv", [str(SCRIPT),
                    "--target_folder", "/tmp/inputs"] + extra):
                self.assertEqual(cli.parse_args().batch_plan_cache_dir, expected)
                self.assertEqual(cli.parse_args().batch_plan_cache_limit, 20)

    def test_removed_flags_are_rejected(self):
        for flag in ("--prebatch", "--use_ordered_files", "--structure_select", "--random_seed"):
            with self.subTest(flag=flag), patch.object(sys, "argv", [str(SCRIPT),
                    "--target_folder", "/tmp/inputs", flag, "true"]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
                cli.parse_args()
            self.assertEqual(exit.exception.code, 2)

    def test_invalid_model_and_empty_input_are_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ("unknown", "mock"):
                with self.subTest(model=model), patch.object(sys, "argv", [str(SCRIPT),
                        "--target_folder", tmp, "--model", model]):
                    self.assertEqual(cli.main(), 1)

    def test_negative_structure_count_fails(self):
        with patch.object(sys, "argv", [str(SCRIPT), "--target_folder", "/tmp/inputs",
                                       "--num_structures", "-1"]):
            self.assertEqual(cli.main(), 1)

    def test_baseline_uses_unified_sample_and_writes_matching_manifest(self):
        from batchase.engine.batching import select_structure_files
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for index in range(10):
                path = root / f"{index:03d}.cif"
                path.write_text("not parsed in rand mode")
                files.append(str(path))
            output = root / "output"
            argv = [str(SCRIPT), "--target_folder", tmp, "--model", "mock",
                    "--num_structures", "3", "--structure_order", "rand",
                    "--structure_order_seed", "17", "--run_baseline",
                    "--output_path", str(output)]
            with patch.object(sys, "argv", argv), patch.object(cli, "run_baseline") as baseline:
                cli.main()
            expected = select_structure_files(files, "rand", 17, 3)
            self.assertEqual(baseline.call_args.kwargs["files"], expected)
            self.assertEqual((output / "manifest.txt").read_text().splitlines(), expected)

    def test_native_launcher_forwards_new_parameters_and_records_them(self):
        project = Path(__file__).resolve().parents[3]
        runner = project / "scripts/relaxation.sh"
        if not runner.exists():
            self.skipTest("Native launcher is only present in the superproject")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copyfile(runner, root / "scripts/relaxation.sh")
            (root / ".venv/bin").mkdir(parents=True)
            os.symlink(sys.executable, root / ".venv/bin/python")
            (root / "3rdparty/batchASE/scripts").mkdir(parents=True)
            (root / "3rdparty/batchASE/scripts/batch_relax.py").write_text(
                "import json, sys; print(json.dumps(sys.argv[1:]))\n")
            config = root / "config.sh"
            config.write_text("""TARGET_FOLDER=/tmp/inputs
MOLECULE_SINGLE=1
N_GPUS=1
GPU_OFFSET=0
NUM_WORKERS=2
NUM_THREADS=1
MAX_STEPS=1
FMAX=0.01
SCALAR_PRESSURE=0
OPTIMIZER1=FIRE
OPTIMIZER2=BFGS
MODEL=mock
BATCH_MODE=atoms
BATCH_SIZE=1
MAX_BATCH_ATOMS=100
STRUCTURE_ORDER=rand
STRUCTURE_ORDER_SEED=17
NUM_STRUCTURES=3
OUTPUT_PATH=output
""")
            result = subprocess.run(["bash", str(root / "scripts/relaxation.sh"), str(config)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads((root / "output/opt.log").read_text())
            for name, value in (("--batch_mode", "atoms"), ("--max_batch_atoms", "100"),
                                ("--structure_order", "rand"), ("--structure_order_seed", "17")):
                self.assertEqual(args[args.index(name) + 1], value)
            self.assertEqual(args[args.index("--batch_plan_cache_dir") + 1],
                             str(root / ".cache/batch_plans"))
            self.assertEqual(args[args.index("--batch_plan_cache_limit") + 1], "20")
            self.assertNotIn("--prebatch", args)
            self.assertNotIn("--use_ordered_files", args)
            self.assertNotIn("--structure_select", args)
            self.assertNotIn("--random_seed", args)
            self.assertEqual(args[args.index("--num_structures") + 1], "3")
            params = (root / "output/run_params.txt").read_text()
            self.assertIn("[Batch Planning]", params)
            self.assertIn("STRUCTURE_ORDER_SEED = 17", params)
            self.assertIn("BATCH_PLAN_CACHE_DIR = " + str(root / ".cache/batch_plans"), params)
            for disabled in ('""', 'none'):
                with self.subTest(disabled=disabled):
                    with config.open("a") as handle:
                        handle.write(f"BATCH_PLAN_CACHE_DIR={disabled}\n")
                    result = subprocess.run(["bash", str(root / "scripts/relaxation.sh"), str(config)],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = json.loads((root / "output/opt.log").read_text())
                    self.assertEqual(args[args.index("--batch_plan_cache_dir") + 1],
                                     "" if disabled == '""' else "none")


if __name__ == "__main__":
    unittest.main()
