from __future__ import annotations

import unittest

import torch

from batchase.relaxation.optimizers.bfgs import BFGS


class QuadraticOptimizable:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.positions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.batch_indices = torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
            device=self.device,
        )
        self._update_mask = torch.ones(2, dtype=torch.bool, device=self.device)
        self.converge_indices_list = []
        self.failed_indices_list = []

    @property
    def batch_size(self) -> int:
        return 2

    @property
    def elem_per_group(self) -> torch.Tensor:
        return torch.bincount(self.batch_indices)

    @property
    def update_mask(self) -> torch.Tensor:
        return self._update_mask

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
        failed = (~torch.isfinite(max_forces)) | (max_forces > f_upper_limit)
        converged = (max_forces < fmax) & (~failed)
        self._update_mask = (~converged) & (~failed)
        self.converge_indices_list = torch.nonzero(
            converged,
            as_tuple=False,
        ).flatten().tolist()
        self.failed_indices_list = torch.nonzero(
            failed,
            as_tuple=False,
        ).flatten().tolist()
        return not self._update_mask.any().item()


class TestCpuBFGS(unittest.TestCase):
    def _make_cpu_optimizer(
        self,
        optimizable: QuadraticOptimizable | None = None,
    ):
        if optimizable is None:
            optimizable = QuadraticOptimizable()
        optimizer = BFGS(
            optimizable,
            maxstep=0.2,
            alpha=70.0,
            bfgs_cpu_thread=2,
        )
        self.addCleanup(optimizer._shutdown_executor)
        self.assertEqual(type(optimizer).__name__, "_BFGSCpu")
        return optimizer, optimizable

    def test_mixed_inactive_and_active_slots_remain_finite(self):
        optimizer, optimizable = self._make_cpu_optimizer()
        optimizable._update_mask = torch.tensor([False, True])

        positions = optimizable.get_positions()
        forces = optimizable.get_forces()
        displacement, lengths = optimizer.prepare_step(positions, forces)
        limited = optimizer.determine_step(displacement, lengths)

        self.assertTrue(torch.isfinite(limited).all().item())
        torch.testing.assert_close(limited[:2], torch.zeros_like(limited[:2]))
        self.assertGreater(torch.linalg.vector_norm(limited[2:], dim=1).max().item(), 0.0)

    def test_all_zero_steps_remain_zero(self):
        optimizer, _ = self._make_cpu_optimizer()
        displacement = torch.zeros((4, 3), dtype=torch.float64)
        lengths = torch.zeros(4, dtype=torch.float64)

        limited = optimizer.determine_step(displacement, lengths)

        self.assertTrue(torch.isfinite(limited).all().item())
        torch.testing.assert_close(limited, torch.zeros_like(limited))

    def test_step_limit_preserves_small_and_scales_large_groups(self):
        optimizer, _ = self._make_cpu_optimizer()
        displacement = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        lengths = torch.linalg.vector_norm(displacement, dim=1)

        limited = optimizer.determine_step(displacement.clone(), lengths)

        torch.testing.assert_close(limited[:2], displacement[:2])
        expected_large = displacement[2:] * 0.04
        torch.testing.assert_close(limited[2:], expected_large)

    def test_threaded_run_uses_default_force_limit(self):
        optimizer, optimizable = self._make_cpu_optimizer()
        initial_force = optimizable.get_max_forces().max().item()

        optimizer.run(fmax=1e-12, steps=5)

        final_force = optimizable.get_max_forces().max().item()
        self.assertIsNotNone(optimizer._executor)
        self.assertEqual(optimizer.nsteps, 5)
        self.assertTrue(torch.isfinite(optimizable.positions).all().item())
        self.assertLess(final_force, initial_force)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_and_gpu_step_limiting_match(self):
        cpu_optimizer, _ = self._make_cpu_optimizer()
        gpu_optimizable = QuadraticOptimizable("cuda:0")
        gpu_optimizer = BFGS(
            gpu_optimizable,
            maxstep=0.2,
            alpha=70.0,
            bfgs_cpu_thread=0,
        )
        displacement = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        lengths = torch.linalg.vector_norm(displacement, dim=1)

        cpu_result = cpu_optimizer.determine_step(
            displacement.clone(),
            lengths,
        )
        gpu_result = gpu_optimizer.determine_step(
            displacement.cuda(),
            lengths.cuda(),
        ).cpu()

        torch.testing.assert_close(cpu_result, gpu_result)


if __name__ == "__main__":
    unittest.main()
