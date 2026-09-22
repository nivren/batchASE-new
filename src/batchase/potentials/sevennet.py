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
        self.kind = "sevennet"
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.r_max = 5.0
        # Placeholder for full SevenNet calculator integration
        self._model_name = model_name

    def build_inputs(self, gbatch) -> dict:
        raise NotImplementedError("SevenNetBatchBackend.build_inputs is under active development.")

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SevenNetBatchBackend.predict is under active development.")

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SevenNetBatchBackend.predict_from_atoms is under active development.")
