"""
Hardware-accelerated kernel registry and dynamic dispatch.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger("batchase.kernels")

_KERNELS = {}


def register_pbc_kernel(name: str):
    def decorator(fn: Callable):
        _KERNELS[name.lower()] = fn
        return fn
    return decorator


def detect_optimal_backend() -> str:
    """Detect the highest-performance operational backend available on this machine."""
    # 1. Check NVIDIA CUDA
    try:
        from .cuda import is_cuda_available
        if is_cuda_available():
            return "cuda"
    except Exception:
        pass

    # 2. Check Hygon DCU / ROCm HIP
    try:
        from .hip import is_hip_available
        if is_hip_available():
            return "hip"
    except Exception:
        pass

    # 3. Check Triton
    try:
        from .triton import is_triton_available
        if is_triton_available():
            return "triton"
    except Exception:
        pass

    # 4. Fallback to CPU PyTorch
    return "cpu"


def get_pbc_graph_kernel(backend: str = "auto") -> Callable:
    """
    Retrieve the PBC radius graph kernel for the requested hardware backend.
    
    Args:
        backend: "auto", "cuda", "hip", "triton", "tilelang", or "cpu".
    """
    backend = backend.lower()
    if backend == "auto":
        backend = detect_optimal_backend()

    if backend == "cuda":
        from .cuda import radius_graph_pbc_cuda
        return radius_graph_pbc_cuda
    elif backend == "hip":
        from .hip import radius_graph_pbc_hip
        return radius_graph_pbc_hip
    elif backend == "triton":
        from .triton import radius_graph_pbc_triton
        return radius_graph_pbc_triton
    elif backend == "cpu":
        from .cpu import radius_graph_pbc_cpu
        return radius_graph_pbc_cpu
    else:
        raise ValueError(f"Unknown kernel backend: {backend}. Available: ['auto', 'cuda', 'hip', 'triton', 'cpu']")
