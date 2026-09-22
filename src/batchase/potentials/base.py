"""
Base protocol and abstract interface for machine learning interatomic potential backends.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
import torch


@runtime_checkable
class BatchPotential(Protocol):
    """
    Standard interface for batched MLIP backends in batchASE.
    
    All MLIP adapters (MACE, SevenNet, CHGNet, MatRIS, etc.) implement this protocol.
    """

    kind: str
    device: torch.device
    dtype: torch.dtype
    r_max: float

    def build_inputs(self, gbatch) -> dict:
        """Construct model forward inputs from a batched PyG graph."""
        ...

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        """
        Batch inference producing standardized outputs:
        - "energy": [B, 1] torch.float64
        - "forces": [N, 3] torch.float64
        - "stress": [B, 3, 3] torch.float64 (optional)
        """
        ...

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        """Predict directly from a sequence of ASE Atoms objects in native float64."""
        ...
