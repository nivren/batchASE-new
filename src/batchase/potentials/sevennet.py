"""
SevenNet batched potential adapter (placeholder/adapter).
"""

from __future__ import annotations

from typing import Sequence
import torch
from .base import BatchPotential


class SevenNetBatchBackend:
    """SevenNet potential adapter implementing BatchPotential."""

    def __init__(self, model_name: str = "7net-0", device: str = "cuda", **kwargs):
        raise NotImplementedError("SevenNetBatchBackend is currently a placeholder under active development.")

    def build_inputs(self, gbatch) -> dict:
        raise NotImplementedError("SevenNetBatchBackend.build_inputs is under active development.")

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SevenNetBatchBackend.predict is under active development.")

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SevenNetBatchBackend.predict_from_atoms is under active development.")
