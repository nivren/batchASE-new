"""Tests for backend registry, supported backend creation, and fail-fast rejection (Task 2.5).

Verifies:
1. create_backend("mace") creates a functional MACEBatchBackend.
2. create_backend() on unimplemented placeholder backends (sevennet, chgnet, matris, matgl)
   immediately raises NotImplementedError with clear guidance.
3. create_backend() on unknown backend raises ValueError.
4. Direct instantiation of stub classes raises NotImplementedError.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from batchase.potentials import (
    SUPPORTED_BACKENDS,
    UNIMPLEMENTED_BACKENDS,
    create_backend,
    MACEBatchBackend,
    SevenNetBatchBackend,
    CHGNetBatchBackend,
    MatRISBatchBackend,
)


class MockMACEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_max = torch.tensor(4.5)


class MockMACECalculator:
    def __init__(self):
        self.models = [MockMACEModel()]
        self.z_table = [1, 6, 8]


class TestBackendRegistry(unittest.TestCase):
    def test_supported_backend_creation(self):
        """create_backend('mace') returns a MACEBatchBackend instance."""
        calc = MockMACECalculator()
        backend = create_backend("mace", calculator=calc, device="cpu")
        self.assertIsInstance(backend, MACEBatchBackend)
        self.assertEqual(backend.kind, "mace")

    def test_unimplemented_backends_fail_fast(self):
        """create_backend on placeholder backends immediately raises NotImplementedError."""
        for name in ("sevennet", "chgnet", "matris", "matgl"):
            with self.subTest(backend=name):
                with self.assertRaises(NotImplementedError) as ctx:
                    create_backend(name)
                err_msg = str(ctx.exception)
                self.assertIn(name, err_msg)
                self.assertIn("mace", err_msg)

    def test_unknown_backend_raises_value_error(self):
        """create_backend on unrecognized backend name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_backend("totally_unknown_mlip_model")
        self.assertIn("totally_unknown_mlip_model", str(ctx.exception))
        self.assertIn("Supported", str(ctx.exception))

    def test_stub_classes_direct_instantiation_raises(self):
        """Direct instantiation of placeholder classes raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            SevenNetBatchBackend()
        with self.assertRaises(NotImplementedError):
            CHGNetBatchBackend()
        with self.assertRaises(NotImplementedError):
            MatRISBatchBackend()


if __name__ == "__main__":
    unittest.main()
