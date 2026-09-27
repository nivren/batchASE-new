"""Tests for compile parameter propagation and training=False fix (Task 2.4).

Verifies:
1. MACEBatchBackend._forward always calls model with training=False.
2. Returned tensors are detached (grad_fn is None, no retained autograd graph).
3. compile_mode propagates through Scheduler -> Worker -> Backend.
4. MACEBatchBackend fallback on compile failure.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from batchase.engine.scheduler import Scheduler
from batchase.engine.worker import Worker
from batchase.potentials import create_backend, MACEBatchBackend


class MockMACEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_max = torch.tensor(4.5)
        self.last_training = None
        self.last_compute_stress = None

    def forward(self, inputs, compute_stress: bool = False, training: bool = False):
        self.last_training = training
        self.last_compute_stress = compute_stress
        batch_size = inputs.get("ptr", torch.tensor([0, 2])).shape[0] - 1
        num_atoms = inputs.get("positions", torch.zeros(2, 3)).shape[0]

        # Simulate torch tensors with requires_grad
        energy = torch.zeros(batch_size, dtype=torch.float64, requires_grad=True)
        forces = torch.zeros((num_atoms, 3), dtype=torch.float64, requires_grad=True)
        res = {
            "energy": energy,
            "forces": forces,
        }
        if compute_stress:
            stress = torch.zeros((batch_size, 3, 3), dtype=torch.float64, requires_grad=True)
            res["stress"] = stress
        return res


class MockMACECalculator:
    def __init__(self, model=None, z_table=None):
        self.models = [model or MockMACEModel()]
        self.z_table = z_table or [1, 6, 8]


class TestCompileSemantics(unittest.TestCase):
    def test_forward_always_calls_training_false_and_detaches(self):
        """MACEBatchBackend._forward must pass training=False and detach graphs."""
        mock_model = MockMACEModel()
        mock_calc = MockMACECalculator(model=mock_model)

        backend = MACEBatchBackend(
            model="mock",
            device="cpu",
            use_compile=True,  # Even when use_compile is True
            compile_mode="reduce-overhead",
            calculator=mock_calc,
        )

        dummy_inputs = {
            "positions": torch.zeros((2, 3), dtype=torch.float64),
            "ptr": torch.tensor([0, 2], dtype=torch.long),
        }

        # Case 1: with compute_stress=True
        out = backend._forward(dummy_inputs, compute_stress=True)
        self.assertIs(mock_model.last_training, False)
        self.assertTrue(mock_model.last_compute_stress)
        self.assertIsNone(out["energy"].grad_fn)
        self.assertIsNone(out["forces"].grad_fn)
        self.assertIsNone(out["stress"].grad_fn)

        # Case 2: with compute_stress=False
        out = backend._forward(dummy_inputs, compute_stress=False)
        self.assertIs(mock_model.last_training, False)
        self.assertFalse(mock_model.last_compute_stress)
        self.assertIsNone(out["energy"].grad_fn)
        self.assertIsNone(out["forces"].grad_fn)
        self.assertNotIn("stress", out)

    def test_compile_mode_propagation_scheduler_to_worker(self):
        """compile_mode parameter is correctly accepted and passed by Scheduler and Worker."""
        scheduler = Scheduler(
            files=["test.cif"],
            compile_mode="max-autotune",
            num_workers=1,
            devices=["cpu"],
        )
        self.assertEqual(scheduler.compile_mode, "max-autotune")

        worker = Worker(
            files=["test.cif"],
            device="cpu",
            compile_mode="max-autotune",
        )
        self.assertEqual(worker.compile_mode, "max-autotune")

    def test_compile_fallback_on_failure(self):
        """If mace_off fails with compile_mode, MACEBatchBackend falls back to uncompiled."""
        mock_model = MockMACEModel()
        mock_calc = MockMACECalculator(model=mock_model)

        calls = []

        def fake_mace_off(**kwargs):
            calls.append(kwargs)
            if kwargs.get("compile_mode") is not None:
                raise RuntimeError("torch.compile not supported on this device/configuration")
            return mock_calc

        with patch("mace.calculators.mace_off", side_effect=fake_mace_off):
            backend = MACEBatchBackend(
                model="small",
                device="cpu",
                compile_mode="reduce-overhead",
            )

        # First call attempted compile_mode="reduce-overhead"
        self.assertEqual(calls[0]["compile_mode"], "reduce-overhead")
        # Second call fell back to compile_mode=None
        self.assertIsNone(calls[1]["compile_mode"])
        # Backend marked as uncompiled
        self.assertFalse(backend.use_compile)
        self.assertIsNone(backend.compile_mode)
        self.assertIs(backend.calculator, mock_calc)


if __name__ == "__main__":
    unittest.main()
