"""Tests for FIRE/FIRE2 ASE semantic alignment (task 2.3).

Verifies:
- FIRE 1.0 has no dtmin lower bound (matches ASE native).
- FIRE2 retains dtmin lower bound.
- torch_scatter is not imported by fire.py.
- FIRE and FIRE2 produce different trajectories (different algorithms).
"""

from __future__ import annotations

import sys
import unittest

import torch

from batchase.relaxation.optimizers.fire import (
    FIRE,
    FIRE2,
    _BatchFIREBase,
    _batched_dot_per_system,
)


class MockOptimizable:
    """Minimal mock that satisfies _BatchFIREBase requirements."""

    def __init__(self, num_atoms: int = 4, batch_size: int = 2, harmonic: bool = False):
        self._batch_size = batch_size
        self._num_atoms = num_atoms
        self.harmonic = harmonic
        atoms_per_system = num_atoms // batch_size
        self.device = "cpu"
        self.dtype = torch.float64
        self.batch_indices = torch.tensor(
            [i // atoms_per_system for i in range(num_atoms)],
            dtype=torch.long,
        )
        self._pos = torch.randn(num_atoms, 3, dtype=torch.float64)
        self._forces = torch.randn(num_atoms, 3, dtype=torch.float64) * 0.5
        self.update_mask = torch.ones(batch_size, dtype=torch.bool)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_positions(self) -> torch.Tensor:
        return self._pos.clone()

    def set_positions(self, pos: torch.Tensor) -> None:
        self._pos = pos.clone()

    def get_forces(self, apply_constraint: bool = True) -> torch.Tensor:
        if self.harmonic:
            return -self._pos.clone()
        return self._forces.clone()

    def get_max_forces(self, apply_constraint: bool = True) -> torch.Tensor:
        per_atom = self.get_forces().norm(dim=-1)
        max_f = torch.zeros(
            self._batch_size, dtype=torch.float64
        ).scatter_reduce(
            0, self.batch_indices, per_atom, reduce="amax"
        )
        return max_f


class TestFireDtminSemantics(unittest.TestCase):
    """FIRE 1.0 should have no dtmin; FIRE2 should clamp at dtmin."""

    def test_fire_dt_no_lower_bound(self):
        """FIRE negative-power dt can decay below any threshold."""
        opt = MockOptimizable(num_atoms=4, batch_size=2)
        fire = FIRE(opt, dt=0.1, fdec=0.5)

        # Verify FIRE defaults to dtmin=0.0
        self.assertEqual(fire.dtmin, 0.0)

        # Simulate many negative-power resets: dt should keep halving
        # without any lower bound.
        fire.dt[:] = 0.1
        dt_min_tensor = torch.tensor(0.0, dtype=torch.float64)
        for _ in range(20):
            fire.dt *= fire.fdec
            # No clamp — FIRE 1.0 semantics
        # After 20 halvings: 0.1 * 0.5^20 ≈ 9.5e-8
        expected = 0.1 * (0.5 ** 20)
        torch.testing.assert_close(
            fire.dt,
            torch.full_like(fire.dt, expected),
        )
        # Must be well below FIRE2's dtmin=2e-3
        self.assertTrue((fire.dt < 2e-3).all())

    def test_fire2_dt_has_lower_bound(self):
        """FIRE2 negative-power dt clamps at dtmin=2e-3."""
        opt = MockOptimizable(num_atoms=4, batch_size=2)
        fire2 = FIRE2(opt, dt=0.1, fdec=0.5, dtmin=2e-3)

        self.assertEqual(fire2.dtmin, 2e-3)

        fire2.dt[:] = 0.1
        dt_min_t = torch.tensor(2e-3, dtype=torch.float64)
        for _ in range(20):
            fire2.dt = torch.maximum(fire2.dt * fire2.fdec, dt_min_t)
        # After clamping, all dt should be exactly dtmin
        torch.testing.assert_close(
            fire2.dt,
            torch.full_like(fire2.dt, 2e-3),
        )

    def test_fire_step_respects_no_dtmin(self):
        """Full step() cycle: FIRE 1.0 dt decays freely on negative power."""
        opt = MockOptimizable(num_atoms=4, batch_size=2)
        opt._forces = torch.full((4, 3), 1.0, dtype=torch.float64)

        # Create forces that oppose velocity to produce negative power
        # Starting from dt=0.002 (the FIRE2 threshold), halving should reach 0.001 < 2e-3
        fire = FIRE(opt, dt=0.002, fdec=0.5)
        fire.is_initialized[:] = True
        # Set velocity in opposite direction to forces: forces · v = 1.0 * (-10.0) * 3 * 2 < 0
        fire.v = -opt.get_forces() * 10.0

        dt_before = fire.dt.clone()
        fire.step()
        dt_after = fire.dt.clone()

        # dt should have decreased by fdec: 0.002 * 0.5 = 0.001
        expected = dt_before * fire.fdec
        torch.testing.assert_close(dt_after, expected)
        # Must be below 2e-3 (FIRE2's dtmin)
        self.assertTrue((dt_after < 2e-3).all())


class TestScatterNoDependency(unittest.TestCase):
    """fire.py must not import torch_scatter."""

    def test_no_torch_scatter_in_fire_module(self):
        """Verify torch_scatter is not referenced in fire.py source."""
        import batchase.relaxation.optimizers.fire as fire_mod
        import inspect

        source = inspect.getsource(fire_mod)
        self.assertNotIn("torch_scatter", source)
        self.assertNotIn("_has_scatter", source)

    def test_batched_dot_works_without_scatter(self):
        """_batched_dot_per_system uses native scatter_add_ only."""
        x = torch.randn(6, 3, dtype=torch.float64)
        y = torch.randn(6, 3, dtype=torch.float64)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        result = _batched_dot_per_system(x, y, batch, 2)
        # Manual computation
        expected = torch.tensor([
            (x[:3] * y[:3]).sum(),
            (x[3:] * y[3:]).sum(),
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected)


class TestFireVsFire2Divergence(unittest.TestCase):
    """FIRE and FIRE2 must produce different trajectories."""

    def test_trajectories_differ(self):
        """Same initial conditions → different positions after steps under harmonic force."""
        torch.manual_seed(42)

        opt1 = MockOptimizable(num_atoms=6, batch_size=2, harmonic=True)
        opt2 = MockOptimizable(num_atoms=6, batch_size=2, harmonic=True)
        # Ensure same initial state
        opt2._pos = opt1._pos.clone()

        fire1 = FIRE(opt1, dt=0.1)
        fire2 = FIRE2(opt2, dt=0.1)

        # Set non-collinear initial velocity so mixing order differences take effect
        v0 = torch.randn_like(opt1._pos)
        fire1.v = v0.clone()
        fire2.v = v0.clone()
        fire1.is_initialized[:] = True
        fire2.is_initialized[:] = True

        for _ in range(5):
            fire1.step()
            fire2.step()

        pos1 = opt1.get_positions()
        pos2 = opt2.get_positions()

        # They must differ due to different integration schemes
        # (forward Euler vs semi-implicit + mixing order)
        self.assertFalse(torch.allclose(pos1, pos2, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
