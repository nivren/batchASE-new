"""
Hygon DCU / AMD ROCm HIP implementation placeholder and wrapper for PBC radius graph.
"""

from __future__ import annotations

import torch
from typing import Optional, List
from ...neighbors.pbc import compute_rep_per_image, sanitize_pbc_flags

_HIP_EXT = None


def is_hip_available() -> bool:
    """Check if ROCm / HIP environment is available."""
    if not torch.cuda.is_available():
        return False
    # Check for ROCm/HIP build
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return True
    return False


def radius_graph_pbc_hip(
    data,
    radius: float,
    pbc: Optional[List[bool]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HIP-accelerated PBC radius graph construction (Hygon DCU / ROCm).
    """
    if not is_hip_available():
        raise NotImplementedError("HIP / ROCm environment is not active or available.")
    raise NotImplementedError("HIP PBC kernel is reserved for DCU deployment.")
