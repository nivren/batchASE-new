from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from batchase.relaxation.optimizers.lbfgs import LBFGS
from batchase.relaxation.optimizers.registry import get_optimizer_cls


class QuadraticOptimizable:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.batch_indices = torch.tensor([0, 0, 1], device=self.device)
        self.batch = SimpleNamespace(
            natoms=torch.tensor([2, 1], device=self.device)
        )
        self.positions = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=self.dtype,
            device=self.device,
        )
        self.results = {}
        self.converge_indices_list = []
        self.failed_indices_list = []

    @property
    def batch_size(self) -> int:
        return 2

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def set_positions(self, positions: torch.Tensor) -> None:
        self.positions = positions.clone()

    def get_forces(self, apply_constraint: bool = False) -> torch.Tensor:
        return -self.positions

    def get_max_forces(
        self,
        forces: torch.Tensor | None = None,
        apply_constraint: bool = False,
    ) -> torch.Tensor:
        if forces is None:
            forces = self.get_forces(apply_constraint=apply_constraint)
        norms = torch.linalg.vector_norm(forces, dim=1)
        return torch.full(
            (self.batch_size,),
            float("-inf"),
            dtype=norms.dtype,
            device=norms.device,
        ).scatter_reduce(
            0,
            self.batch_indices,
            norms,
            reduce="amax",
            include_self=True,
        )

    def converged(
        self,
        forces: torch.Tensor | None = None,
        fmax: float = 0.05,
        max_forces: torch.Tensor | None = None,
        f_upper_limit: float = 100.0,
    ) -> bool:
        if max_forces is None:
            max_forces = self.get_max_forces(forces=forces)
        self.converge_indices_list = torch.nonzero(
            max_forces < fmax,
            as_tuple=False,
        ).flatten().tolist()
        return len(self.converge_indices_list) == self.batch_size


class TestLBFGS(unittest.TestCase):
    def _make_optimizer(self, device: str) -> tuple[LBFGS, QuadraticOptimizable]:
        optimizable = QuadraticOptimizable(device)
        optimizer_cls = get_optimizer_cls("LBFGS")
        optimizer = optimizer_cls(
            optimizable,
            maxstep=0.2,
            damping=1.0,
            early_stop=False,
            f_upper_limit=100.0,
        )
        return optimizer, optimizable

    def _assert_helper_results(self, device: str) -> None:
        optimizer, _ = self._make_optimizer(device)
        displacement = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        limited = optimizer.determine_step(displacement.clone())
        self.assertTrue(torch.isfinite(limited).all().item())
        expected_lengths = torch.tensor(
            [0.0, 0.2, 0.1],
            dtype=torch.float64,
            device=device,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(limited, dim=1),
            expected_lengths,
        )

        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float64,
            device=device,
        )
        y = torch.tensor(
            [[2.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            dtype=torch.float64,
            device=device,
        )
        expected_dot = torch.tensor(
            [15.0, 28.0],
            dtype=torch.float64,
            device=device,
        )
        torch.testing.assert_close(optimizer._batched_dot(x, y), expected_dot)

    def test_cpu_helpers_and_zero_step(self):
        self._assert_helper_results("cpu")

    def test_cpu_run_five_steps(self):
        optimizer, optimizable = self._make_optimizer("cpu")
        initial_max = optimizable.get_max_forces().max().item()
        optimizer.run(fmax=1e-12, steps=5)
        final_max = optimizable.get_max_forces().max().item()

        self.assertEqual(optimizer.nsteps, 5)
        self.assertTrue(torch.isfinite(optimizable.positions).all().item())
        self.assertLess(final_max, initial_max)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_helpers_and_zero_step(self):
        self._assert_helper_results("cuda:0")


if __name__ == "__main__":
    unittest.main()
