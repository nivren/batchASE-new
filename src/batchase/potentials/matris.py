"""
MatRIS / MatGL batched potential adapter (placeholder/adapter).
"""

from __future__ import annotations

from typing import Sequence
import torch
from .base import BatchPotential


class MatRISBatchBackend:
    """MatRIS / MatGL potential adapter implementing BatchPotential."""

    def __init__(self, model_name: str = "default", device: str = "cuda", **kwargs):
        self.kind = "matris"
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.r_max = 5.0
        self._model_name = model_name

    def build_inputs(self, gbatch) -> dict:
        raise NotImplementedError("MatRISBatchBackend.build_inputs is under active development.")

    def predict(self, gbatch, compute_stress: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError("MatRISBatchBackend.predict is under active development.")

    def predict_from_atoms(
        self, atoms_list: Sequence, compute_stress: bool = False
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("MatRISBatchBackend.predict_from_atoms is under active development.")
