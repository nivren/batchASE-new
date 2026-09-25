"""
Unified Linear Algebra execution strategy and device dispatch.

Provides unified batch matrix operations (such as eigenvalue decomposition `eigh`),
allowing dynamic offload between GPU (CUDA/DCU) and CPU (multithreaded MKL/OpenBLAS)
without duplicating optimizer code.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union
import torch

from .cusolver_batched import cusolver_syevj_batched, is_cusolver_batched_available

logger = logging.getLogger("batchase.relaxation.linalg")


class LinalgBackend:
    """
    Linear algebra execution backend for batched optimization.
    
    Supports:
    - "cuda" / "gpu": direct GPU execution via torch.linalg.
    - "cpu": offload matrix decomposition to CPU with configurable thread pool,
      which is often faster for small crystal cell matrices (3N <= 300) or on DCU.
    - "auto": automatic selection based on dimension and hardware.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "auto",
        threads: int = 1,
    ) -> None:
        self.requested_device = str(device).lower()
        self.threads = threads
        self._executor = None
        if self.requested_device == "cpu" and self.threads > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.threads)

    def robust_eigh(
        self, H: torch.Tensor, target_device: Optional[Union[str, torch.device]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenvalue decomposition of symmetric batch matrix H: [B, D, D].
        
        Args:
            H: [B, D, D] symmetric tensor.
            target_device: device override for this operation ("cpu", "cuda", etc.).
            
        Returns:
            eigenvalues: [B, D]
            eigenvectors: [B, D, D]
        """
        original_device = H.device
        exec_dev = str(target_device or self.requested_device).lower()

        if exec_dev == "auto":
            # If batch has small degrees of freedom (e.g. D <= 180) and CPU threads available,
            # CPU BLAS can be competitive; otherwise use GPU.
            exec_dev = "cpu" if (H.shape[-1] <= 150 and self.threads > 4) else ("cuda" if H.is_cuda else "cpu")

        if exec_dev == "cpu" and H.is_cuda:
            H_cpu = H.detach().cpu()
            if self._executor is not None and H.shape[0] > 1:
                # Parallel eigh across batch items on CPU thread pool
                def _eigh_single(item):
                    return torch.linalg.eigh(item)

                futures = [self._executor.submit(_eigh_single, H_cpu[i]) for i in range(H_cpu.shape[0])]
                results = [f.result() for f in futures]
                vals = torch.stack([r[0] for r in results], dim=0).to(original_device)
                vecs = torch.stack([r[1] for r in results], dim=0).to(original_device)
                return vals, vecs
            else:
                vals, vecs = torch.linalg.eigh(H_cpu)
                return vals.to(original_device), vecs.to(original_device)
        else:
            if H.is_cuda and is_cusolver_batched_available() and H.dim() == 3 and H.shape[-1] > 32:
                return cusolver_syevj_batched(H)
            return torch.linalg.eigh(H)
