"""
CHGNet batched potential adapter (placeholder/adapter).
"""

from __future__ import annotations

from typing import Sequence
import torch
from .base import BatchPotential


class CHGNetBatchBackend:
    """CHGNet potential adapter implementing BatchPotential."""

    def __init__(self, model_name: str = "default", device: str = "cuda", **kwargs):
        raise NotImplementedError("CHGNetBatchBackend is currently a placeholder under active development.")

    def build_inputs(self, gbatch) -> dict:
        raise NotImplementedError("CHGNetBatchBackend.build_inputs is under active development.")

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError("CHGNetBatchBackend.predict is under active development.")

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("CHGNetBatchBackend.predict_from_atoms is under active development.")
