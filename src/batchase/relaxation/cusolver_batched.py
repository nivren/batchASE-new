"""
Native cuSOLVER batched eigenvalue decomposition (syevjBatched) via ctypes.

Provides direct access to NVIDIA cuSOLVER's native `cusolverDnDsyevjBatched` (float64)
and `cusolverDnSsyevjBatched` (float32) for batched symmetric matrices, bypassing
PyTorch's upstream n <= 32 gate and achieving massive speedups for crystal degree-of-freedom
matrices (e.g. 3N = 150 ~ 600).
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, Optional, Tuple
import torch

logger = logging.getLogger("batchase.relaxation.cusolver_batched")

# cuSOLVER Constants
CUSOLVER_STATUS_SUCCESS = 0
CUSOLVER_EIG_MODE_NOVECTOR = 0
CUSOLVER_EIG_MODE_VECTOR = 1
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1

_CUSOLVER_LIB = None
_IS_AVAILABLE = False


def _find_and_load_cusolver() -> Optional[ctypes.CDLL]:
    """Search and load libcusolver dynamic library."""
    lib_names = [
        "libcusolver.so.12",
        "libcusolver.so.11",
        "libcusolver.so",
    ]
    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            return lib
        except OSError:
            continue
    return None


try:
    _CUSOLVER_LIB = _find_and_load_cusolver()
    if _CUSOLVER_LIB is not None:
        # Define argument types for syevj functions
        _CUSOLVER_LIB.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _CUSOLVER_LIB.cusolverDnCreate.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnDestroy.argtypes = [ctypes.c_void_p]
        _CUSOLVER_LIB.cusolverDnDestroy.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _CUSOLVER_LIB.cusolverDnSetStream.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnCreateSyevjInfo.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _CUSOLVER_LIB.cusolverDnCreateSyevjInfo.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnDestroySyevjInfo.argtypes = [ctypes.c_void_p]
        _CUSOLVER_LIB.cusolverDnDestroySyevjInfo.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnXsyevjSetTolerance.argtypes = [ctypes.c_void_p, ctypes.c_double]
        _CUSOLVER_LIB.cusolverDnXsyevjSetTolerance.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnXsyevjSetMaxSweeps.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _CUSOLVER_LIB.cusolverDnXsyevjSetMaxSweeps.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnXsyevjSetSortEig.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _CUSOLVER_LIB.cusolverDnXsyevjSetSortEig.restype = ctypes.c_int

        # Double precision (float64)
        _CUSOLVER_LIB.cusolverDnDsyevjBatched_bufferSize.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int
        ]
        _CUSOLVER_LIB.cusolverDnDsyevjBatched_bufferSize.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnDsyevjBatched.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int
        ]
        _CUSOLVER_LIB.cusolverDnDsyevjBatched.restype = ctypes.c_int

        # Single precision (float32)
        _CUSOLVER_LIB.cusolverDnSsyevjBatched_bufferSize.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int
        ]
        _CUSOLVER_LIB.cusolverDnSsyevjBatched_bufferSize.restype = ctypes.c_int

        _CUSOLVER_LIB.cusolverDnSsyevjBatched.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int
        ]
        _CUSOLVER_LIB.cusolverDnSsyevjBatched.restype = ctypes.c_int

        _IS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Failed to initialize cuSOLVER ctypes wrapper: {e}")
    _IS_AVAILABLE = False


def is_cusolver_batched_available() -> bool:
    """Return whether native cuSOLVER syevjBatched is available on this system."""
    return _IS_AVAILABLE and torch.cuda.is_available()


class _DeviceContext:
    """Per-device cuSOLVER handle and workspace state."""

    def __init__(self, device: torch.device):
        self.device = device
        self.handle = ctypes.c_void_p()
        self.params = ctypes.c_void_p()
        self.workspace_cache: Dict[Tuple[torch.dtype, int], torch.Tensor] = {}

        with torch.cuda.device(device):
            status = _CUSOLVER_LIB.cusolverDnCreate(ctypes.byref(self.handle))
            if status != CUSOLVER_STATUS_SUCCESS:
                raise RuntimeError(f"cusolverDnCreate failed with status {status}")

            status = _CUSOLVER_LIB.cusolverDnCreateSyevjInfo(ctypes.byref(self.params))
            if status != CUSOLVER_STATUS_SUCCESS:
                raise RuntimeError(f"cusolverDnCreateSyevjInfo failed with status {status}")

            # Configure syevj solver: tol=1e-12 (float64) / 1e-7 (float32), sweeps=100, sort=True
            _CUSOLVER_LIB.cusolverDnXsyevjSetTolerance(self.params, ctypes.c_double(1e-12))
            _CUSOLVER_LIB.cusolverDnXsyevjSetMaxSweeps(self.params, ctypes.c_int(100))
            _CUSOLVER_LIB.cusolverDnXsyevjSetSortEig(self.params, ctypes.c_int(1))

    def __del__(self):
        try:
            if self.params and self.params.value:
                _CUSOLVER_LIB.cusolverDnDestroySyevjInfo(self.params)
            if self.handle and self.handle.value:
                _CUSOLVER_LIB.cusolverDnDestroy(self.handle)
        except Exception:
            pass


_DEVICE_CONTEXTS: Dict[int, _DeviceContext] = {}


def _get_device_context(device: torch.device) -> _DeviceContext:
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    if dev_idx not in _DEVICE_CONTEXTS:
        _DEVICE_CONTEXTS[dev_idx] = _DeviceContext(torch.device("cuda", dev_idx))
    return _DEVICE_CONTEXTS[dev_idx]


def cusolver_syevj_batched(
    A: torch.Tensor,
    tolerance: Optional[float] = None,
    max_sweeps: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute batched eigenvalue decomposition of symmetric 3D tensor A [B, N, N].

    Args:
        A: [B, N, N] symmetric tensor on CUDA (float64 or float32).
        tolerance: Convergence tolerance for Jacobi sweeps.
        max_sweeps: Maximum Jacobi iterations.

    Returns:
        eigenvalues: [B, N] sorted in ascending order.
        eigenvectors: [B, N, N] where column k (vecs[:, :, k]) is eigenvector k.
    """
    if not is_cusolver_batched_available() or not A.is_cuda:
        # Fallback to PyTorch eigh
        return torch.linalg.eigh(A)

    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Input tensor A must have shape [B, N, N], got {A.shape}")

    B, N = A.shape[0], A.shape[1]
    if B == 0 or N == 0:
        return torch.empty((B, N), dtype=A.dtype, device=A.device), torch.empty_like(A)

    ctx = _get_device_context(A.device)
    stream = torch.cuda.current_stream(A.device).cuda_stream
    _CUSOLVER_LIB.cusolverDnSetStream(ctx.handle, ctypes.c_void_p(stream))

    dtype = A.dtype
    A_work = A.contiguous().clone()  # cuSOLVER overwrites input matrix with eigenvectors
    W = torch.empty((B, N), dtype=dtype, device=A.device)
    info = torch.zeros(B, dtype=torch.int32, device=A.device)
    lwork = ctypes.c_int()

    if tolerance is None:
        effective_tol = 1e-12 if dtype == torch.float64 else 1e-7
    else:
        effective_tol = float(tolerance)

    _CUSOLVER_LIB.cusolverDnXsyevjSetTolerance(ctx.params, ctypes.c_double(effective_tol))
    _CUSOLVER_LIB.cusolverDnXsyevjSetMaxSweeps(ctx.params, ctypes.c_int(max_sweeps))

    if dtype == torch.float64:
        status = _CUSOLVER_LIB.cusolverDnDsyevjBatched_bufferSize(
            ctx.handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            N,
            ctypes.c_void_p(A_work.data_ptr()),
            N,
            ctypes.c_void_p(W.data_ptr()),
            ctypes.byref(lwork),
            ctx.params,
            B,
        )
        if status != CUSOLVER_STATUS_SUCCESS:
            raise RuntimeError(f"cusolverDnDsyevjBatched_bufferSize failed with code {status}")

        # Note: lwork.value is count of doubles, not bytes
        cache_key = (dtype, lwork.value)
        workspace = ctx.workspace_cache.get(cache_key)
        if workspace is None or workspace.numel() < lwork.value:
            workspace = torch.empty(lwork.value, dtype=dtype, device=A.device)
            ctx.workspace_cache[cache_key] = workspace

        status = _CUSOLVER_LIB.cusolverDnDsyevjBatched(
            ctx.handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            N,
            ctypes.c_void_p(A_work.data_ptr()),
            N,
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            lwork.value,
            ctypes.c_void_p(info.data_ptr()),
            ctx.params,
            B,
        )
        if status != CUSOLVER_STATUS_SUCCESS:
            raise RuntimeError(f"cusolverDnDsyevjBatched failed with code {status}")

    elif dtype == torch.float32:
        status = _CUSOLVER_LIB.cusolverDnSsyevjBatched_bufferSize(
            ctx.handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            N,
            ctypes.c_void_p(A_work.data_ptr()),
            N,
            ctypes.c_void_p(W.data_ptr()),
            ctypes.byref(lwork),
            ctx.params,
            B,
        )
        if status != CUSOLVER_STATUS_SUCCESS:
            raise RuntimeError(f"cusolverDnSsyevjBatched_bufferSize failed with code {status}")

        cache_key = (dtype, lwork.value)
        workspace = ctx.workspace_cache.get(cache_key)
        if workspace is None or workspace.numel() < lwork.value:
            workspace = torch.empty(lwork.value, dtype=dtype, device=A.device)
            ctx.workspace_cache[cache_key] = workspace

        status = _CUSOLVER_LIB.cusolverDnSsyevjBatched(
            ctx.handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            N,
            ctypes.c_void_p(A_work.data_ptr()),
            N,
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            lwork.value,
            ctypes.c_void_p(info.data_ptr()),
            ctx.params,
            B,
        )
        if status != CUSOLVER_STATUS_SUCCESS:
            raise RuntimeError(f"cusolverDnSsyevjBatched failed with code {status}")
    else:
        raise TypeError(f"cusolver_syevj_batched only supports float32 and float64, got {dtype}")

    # In cuSOLVER column-major layout, eigenvectors are stored in columns.
    # In PyTorch C-contiguous row-major layout, this corresponds to A_work.transpose(-1, -2).
    # Thus V = A_work.transpose(-1, -2) yields column-vector eigenvectors matching torch.linalg.eigh.
    V = A_work.transpose(-1, -2)
    return W, V
