"""
Base class and standard contract for batched structure optimizers.
"""

from __future__ import annotations

from typing import List, Optional
import torch

from ..optimizable import OptimizableBatch


class BatchOptimizer:
    """
    Abstract base class for all batch crystal/molecular structure optimizers.
    
    Subclasses must implement:
    - step(): advance one optimization step for all active systems in the batch.
    - run(fmax, steps): execute relaxation until convergence or max steps.
    - update_slots(keep_indices, num_new_slots): update internal optimizer state
      when Dynamic Slot Management modifies the batch composition.
    """

    def __init__(
        self,
        optimizable: OptimizableBatch,
        fmax: float = 0.01,
        maxstep: float = 0.2,
    ) -> None:
        self.optimizable = optimizable
        self.fmax = fmax
        self.maxstep = maxstep
        self.nsteps: int = 0
        self.device = optimizable.device

    def step(self, fmax: float = 0.01) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError

    def run(self, fmax: float = 0.01, steps: int = 100) -> List[int]:
        """
        Run the optimizer until convergence or steps is reached.
        
        Returns:
            converged_indices: List of batch indices that achieved convergence.
        """
        raise NotImplementedError

    def update_slots(self, keep_indices: List[int], num_new_slots: int) -> None:
        """
        Update internal optimizer states (e.g. Hessian, velocities, history)
        when the batch is dynamically compacted or filled with new structures.
        
        Args:
            keep_indices: Indices of structures retained from the previous batch.
            num_new_slots: Number of newly introduced structures appended to the batch.
        """
        pass
