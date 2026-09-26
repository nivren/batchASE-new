"""
Comprehensive unit tests for active/converged/failed three-state management,
realistic force thresholds, inverted cells, terminal irreversibility, and CUDA vectorized path.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch
from ase import Atoms

from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater
from batchase.relaxation import (
    OptimizableBatch,
    OptimizableUnitCellBatch,
    SlotStatus,
    FailReason,
)


class DummyMockBackend:
    def __init__(self, device="cpu", dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.kind = "mock"

    def predict(self, batch, compute_stress=False):
        n_atoms = len(batch.pos)
        batch_size = len(batch) if hasattr(batch, "__len__") else getattr(batch, "num_graphs", 1)
        res = {
            "energy": torch.zeros(batch_size, dtype=self.dtype, device=self.device),
            "forces": torch.zeros((n_atoms, 3), dtype=self.dtype, device=self.device),
        }
        if compute_stress:
            res["stress"] = torch.zeros((batch_size, 3, 3), dtype=self.dtype, device=self.device)
        return res


def create_mock_batch(n_structures: int = 4, device: str = "cpu"):
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    atoms_list = []
    for _ in range(n_structures):
        a = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]], pbc=True)
        atoms_list.append(a)
    graphs = [a2g.convert(a) for a in atoms_list]
    gbatch = data_list_collater(graphs).to(device)
    return gbatch


class TestSlotStates(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.backend = DummyMockBackend(device=self.device)

    def test_three_states_nan_inf_overflow(self):
        """Verify that NaN, Inf, and realistic overflow forces (150 eV/A with limit 100) are classified as FAILED."""
        gbatch = create_mock_batch(n_structures=4, device=self.device)
        obatch = OptimizableBatch(gbatch, backend=self.backend, dtype=torch.float64)

        # Slot 0: NaN force -> FAILED (nan_force)
        # Slot 1: Inf force -> FAILED (inf_force)
        # Slot 2: 150.0 force (> realistic limit 100.0) -> FAILED (force_overflow)
        # Slot 3: 0.01 force (< fmax 0.05) -> CONVERGED
        max_forces = torch.tensor([float("nan"), float("inf"), 150.0, 0.01], dtype=torch.float64)
        
        is_all_finished = obatch.converged(
            forces=None,
            fmax=0.05,
            max_forces=max_forces,
            f_upper_limit=100.0,
        )

        self.assertTrue(is_all_finished)
        self.assertEqual(obatch.active_indices_list, [])
        self.assertEqual(obatch.converge_indices_list, [3])
        self.assertEqual(sorted(obatch.failed_indices_list), [0, 1, 2])

        reasons = obatch.failed_reasons
        self.assertEqual(reasons[0], FailReason.NAN_FORCE)
        self.assertEqual(reasons[1], FailReason.INF_FORCE)
        self.assertEqual(reasons[2], FailReason.FORCE_OVERFLOW)
        self.assertFalse(torch.any(obatch.update_mask).item())

    def test_zero_and_inverted_cell_detection(self):
        """Verify that zero-volume cells and inverted cells (det < 0) are both detected as invalid_cell."""
        gbatch = create_mock_batch(n_structures=3, device=self.device)
        # Slot 0: Normal valid cell
        # Slot 1: Collapsed cell (zero volume, det == 0)
        gbatch.cell[1] = torch.zeros((3, 3), dtype=torch.float64)
        # Slot 2: Inverted cell with negative determinant (det < 0)
        inverted_cell = torch.tensor([
            [5.43, 0.0, 0.0],
            [0.0, 5.43, 0.0],
            [0.0, 0.0, -5.43]  # flipped z-axis yields negative det
        ], dtype=torch.float64)
        gbatch.cell[2] = inverted_cell

        obatch = OptimizableUnitCellBatch(gbatch, backend=self.backend, dtype=torch.float64)

        max_forces = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        is_all_finished = obatch.converged(
            forces=None,
            fmax=0.05,
            max_forces=max_forces,
            f_upper_limit=100.0,
        )

        self.assertTrue(is_all_finished)
        # Only slot 0 has valid cell and converged force
        self.assertEqual(obatch.converge_indices_list, [0])
        # Both slot 1 and slot 2 must fail due to invalid_cell
        self.assertEqual(sorted(obatch.failed_indices_list), [1, 2])
        self.assertEqual(obatch.failed_reasons[1], FailReason.INVALID_CELL)
        self.assertEqual(obatch.failed_reasons[2], FailReason.INVALID_CELL)

    def test_terminal_irreversibility(self):
        """Verify that once a slot enters CONVERGED or FAILED, it never reverts to ACTIVE."""
        gbatch = create_mock_batch(n_structures=3, device=self.device)
        obatch = OptimizableBatch(gbatch, backend=self.backend, dtype=torch.float64, mask_converged=True)

        # Step 1: Slot 0 converges, Slot 1 fails (overflow), Slot 2 remains active
        forces_step1 = torch.tensor([0.01, 200.0, 0.20], dtype=torch.float64)
        obatch.converged(forces=None, fmax=0.05, max_forces=forces_step1, f_upper_limit=100.0)

        self.assertEqual(obatch.slot_status[0], SlotStatus.CONVERGED)
        self.assertEqual(obatch.slot_status[1], SlotStatus.FAILED)
        self.assertEqual(obatch.slot_status[2], SlotStatus.ACTIVE)

        # Step 2: Slot 0 force spikes to 5.0 (model noise), Slot 1 drops to 0.01
        forces_step2 = torch.tensor([5.0, 0.01, 0.02], dtype=torch.float64)
        obatch.converged(forces=None, fmax=0.05, max_forces=forces_step2, f_upper_limit=100.0)

        # Slot 0 must remain CONVERGED; Slot 1 must remain FAILED; Slot 2 now CONVERGED
        self.assertEqual(obatch.slot_status[0], SlotStatus.CONVERGED)
        self.assertEqual(obatch.slot_status[1], SlotStatus.FAILED)
        self.assertEqual(obatch.slot_status[2], SlotStatus.CONVERGED)
        self.assertEqual(sorted(obatch.converge_indices_list), [0, 2])
        self.assertEqual(obatch.failed_indices_list, [1])

    def test_cuda_vectorized_path(self):
        """Verify tensorized vectorization on CUDA gives exact parity with CPU logic."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        dev = "cuda:0"
        backend_cuda = DummyMockBackend(device=dev)
        gbatch_cuda = create_mock_batch(n_structures=4, device=dev)
        # Slot 1 inverted cell
        gbatch_cuda.cell[1, 2, 2] = -5.43

        obatch_cuda = OptimizableUnitCellBatch(gbatch_cuda, backend=backend_cuda, dtype=torch.float64)

        # Slot 0: 0.01 -> CONVERGED
        # Slot 1: Inverted cell -> FAILED (invalid_cell)
        # Slot 2: 120.0 -> FAILED (force_overflow)
        # Slot 3: NaN -> FAILED (nan_force)
        max_forces = torch.tensor([0.01, 0.01, 120.0, float("nan")], device=dev, dtype=torch.float64)

        all_done = obatch_cuda.converged(
            forces=None,
            fmax=0.05,
            max_forces=max_forces,
            f_upper_limit=100.0,
        )

        self.assertTrue(all_done)
        self.assertEqual(obatch_cuda.converge_indices_list, [0])
        self.assertEqual(sorted(obatch_cuda.failed_indices_list), [1, 2, 3])
        self.assertEqual(obatch_cuda.failed_reasons[1], FailReason.INVALID_CELL)
        self.assertEqual(obatch_cuda.failed_reasons[2], FailReason.FORCE_OVERFLOW)
        self.assertEqual(obatch_cuda.failed_reasons[3], FailReason.NAN_FORCE)

    def test_mark_failed_explicit(self):
        """Verify explicit mark_failed method."""
        gbatch = create_mock_batch(n_structures=2, device=self.device)
        obatch = OptimizableBatch(gbatch, backend=self.backend, dtype=torch.float64)

        obatch.mark_failed(1, FailReason.EIGENSOLVER_FAILED)

        self.assertEqual(obatch.slot_status[1], SlotStatus.FAILED)
        self.assertEqual(obatch.failed_reasons[1], FailReason.EIGENSOLVER_FAILED)
        self.assertEqual(obatch.failed_indices_list, [1])
        self.assertFalse(obatch.update_mask[1].item())


if __name__ == "__main__":
    unittest.main()
