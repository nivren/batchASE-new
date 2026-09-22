"""
CUDA implementation wrapper for outer-product PBC radius graph.
"""

from __future__ import annotations

import os
from typing import Optional, List
import torch

from ...neighbors.pbc import compute_rep_per_image, sanitize_pbc_flags

# Global cache for compiled/loaded CUDA extension
_CUDA_EXT = None


def get_cuda_extension():
    """Load or retrieve the CUDA extension."""
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT

    try:
        import pbc_graph_outer_cuda as ext
        _CUDA_EXT = ext
        return _CUDA_EXT
    except ImportError:
        pass

    # Try JIT compilation fallback if source files exist
    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_src = os.path.join(current_dir, "pbc_graph_outer_wrapper.cpp")
        cu_src = os.path.join(current_dir, "pbc_graph_outer.cu")
        if os.path.exists(cpp_src) and os.path.exists(cu_src):
            _CUDA_EXT = load(
                name="batchase_pbc_outer_cuda",
                sources=[cpp_src, cu_src],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
                verbose=False,
            )
            return _CUDA_EXT
    except Exception as e:
        raise ImportError(f"Failed to load or compile CUDA extension: {e}")

    raise ImportError("pbc_graph_outer_cuda extension is not installed.")


def is_cuda_available() -> bool:
    """Check if CUDA and the CUDA extension are operational."""
    if not torch.cuda.is_available():
        return False
    try:
        get_cuda_extension()
        return True
    except Exception:
        return False


def radius_graph_pbc_cuda(
    data,
    radius: float,
    pbc: Optional[List[bool]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Outer-product / tiling-based CUDA-accelerated PBC radius graph construction.
    """
    ext = get_cuda_extension()
    device = data.pos.device
    pbc = sanitize_pbc_flags(data, pbc)

    # Compute image offsets: [0, natoms[0], natoms[0]+natoms[1], ...]
    img_offset = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(data.natoms, dim=0)
    ])

    rep = compute_rep_per_image(data.cell, radius, pbc, dtype)

    edge_index, unit_cell, num_neighbors_image = ext.radius_graph_pbc_outer_cuda(
        data.pos,
        data.natoms,
        img_offset,
        data.cell,
        rep,
        float(radius),
    )

    return edge_index, unit_cell, num_neighbors_image
