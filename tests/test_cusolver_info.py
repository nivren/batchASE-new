from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from batchase.relaxation.cusolver_batched import (
    _recover_failed_slices,
    cusolver_syevj_batched,
    is_cusolver_batched_available,
)


def make_symmetric_batch(
    batch_size: int = 3,
    dimension: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    torch.manual_seed(7)
    source = torch.randn(
        batch_size,
        dimension,
        dimension,
        dtype=torch.float64,
        device=device,
    )
    return source + source.transpose(-1, -2)


class TestCuSolverInfoRecovery(unittest.TestCase):
    def test_success_path_does_not_call_fallback(self):
        matrices = make_symmetric_batch()
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
        info = torch.zeros(3, dtype=torch.int32)

        with patch(
            "batchase.relaxation.cusolver_batched.torch.linalg.eigh",
            side_effect=AssertionError("fallback should not run"),
        ) as fallback:
            result_values, result_vectors = _recover_failed_slices(
                matrices,
                eigenvalues.clone(),
                eigenvectors.clone(),
                info,
            )

        fallback.assert_not_called()
        torch.testing.assert_close(result_values, eigenvalues)
        torch.testing.assert_close(result_vectors, eigenvectors)

    def test_info_code_recovers_only_failed_slice(self):
        matrices = make_symmetric_batch()
        dimension = matrices.shape[-1]
        eigenvalues = torch.zeros(3, dimension, dtype=torch.float64)
        eigenvectors = torch.eye(dimension, dtype=torch.float64).repeat(3, 1, 1)
        info = torch.tensor([0, dimension + 1, 0], dtype=torch.int32)
        original_eigh = torch.linalg.eigh

        with self.assertLogs(
            "batchase.relaxation.cusolver_batched",
            level="WARNING",
        ) as logs, patch(
            "batchase.relaxation.cusolver_batched.torch.linalg.eigh",
            wraps=original_eigh,
        ) as fallback:
            result_values, result_vectors = _recover_failed_slices(
                matrices,
                eigenvalues.clone(),
                eigenvectors.clone(),
                info,
            )

        fallback.assert_called_once()
        fallback_input = fallback.call_args.args[0]
        torch.testing.assert_close(fallback_input, matrices[1:2])
        expected_values, expected_vectors = original_eigh(matrices[1:2])
        torch.testing.assert_close(result_values[1:2], expected_values)
        torch.testing.assert_close(result_vectors[1:2], expected_vectors)
        torch.testing.assert_close(result_values[[0, 2]], eigenvalues[[0, 2]])
        torch.testing.assert_close(result_vectors[[0, 2]], eigenvectors[[0, 2]])
        self.assertIn("indices=[1]", "\n".join(logs.output))
        self.assertIn(f"info=[{dimension + 1}]", "\n".join(logs.output))

    def test_nonfinite_output_recovers_even_with_zero_info(self):
        matrices = make_symmetric_batch()
        dimension = matrices.shape[-1]
        eigenvalues = torch.zeros(3, dimension, dtype=torch.float64)
        eigenvalues[2, 0] = float("nan")
        eigenvectors = torch.eye(dimension, dtype=torch.float64).repeat(3, 1, 1)
        info = torch.zeros(3, dtype=torch.int32)

        with self.assertLogs(
            "batchase.relaxation.cusolver_batched",
            level="WARNING",
        ) as logs:
            result_values, result_vectors = _recover_failed_slices(
                matrices,
                eigenvalues,
                eigenvectors,
                info,
            )

        self.assertTrue(torch.isfinite(result_values).all().item())
        self.assertTrue(torch.isfinite(result_vectors).all().item())
        self.assertIn("nonfinite_indices=[2]", "\n".join(logs.output))

    def test_fallback_failure_raises_with_slice_context(self):
        matrices = make_symmetric_batch()
        dimension = matrices.shape[-1]
        eigenvalues = torch.zeros(3, dimension, dtype=torch.float64)
        eigenvectors = torch.eye(dimension, dtype=torch.float64).repeat(3, 1, 1)
        info = torch.tensor([0, dimension + 1, 0], dtype=torch.int32)

        with self.assertLogs(
            "batchase.relaxation.cusolver_batched",
            level="ERROR",
        ), patch(
            "batchase.relaxation.cusolver_batched.torch.linalg.eigh",
            side_effect=RuntimeError("forced fallback failure"),
        ), self.assertRaisesRegex(
            RuntimeError,
            rf"indices=\[1\], info=\[{dimension + 1}\]",
        ):
            _recover_failed_slices(
                matrices,
                eigenvalues,
                eigenvectors,
                info,
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and is_cusolver_batched_available(),
        "native cuSOLVER batched path unavailable",
    )
    def test_native_cusolver_normal_batch_residual(self):
        matrices = make_symmetric_batch(
            batch_size=4,
            dimension=64,
            device="cuda:0",
        )

        eigenvalues, eigenvectors = cusolver_syevj_batched(matrices)

        residual = matrices @ eigenvectors - eigenvectors * eigenvalues.unsqueeze(-2)
        relative_residual = torch.linalg.matrix_norm(residual) / torch.clamp(
            torch.linalg.matrix_norm(matrices),
            min=1e-30,
        )
        self.assertLess(relative_residual.max().item(), 1e-11)

    @unittest.skipUnless(
        torch.cuda.is_available() and is_cusolver_batched_available(),
        "native cuSOLVER batched path unavailable",
    )
    def test_native_nonfinite_input_raises_clear_error(self):
        matrices = make_symmetric_batch(
            batch_size=2,
            dimension=16,
            device="cuda:0",
        )
        matrices[1, 0, 0] = float("nan")

        with self.assertRaisesRegex(RuntimeError, r"fallback failed.*indices=\[1\]"):
            cusolver_syevj_batched(matrices)


if __name__ == "__main__":
    unittest.main()
