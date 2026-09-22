"""
PBC utilities and replication calculation for periodic crystal structures.
"""

from __future__ import annotations

import torch
from typing import List, Optional


def compute_rep_per_image(
    cell: torch.Tensor,
    radius: float,
    pbc: List[bool],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute independent PBC replication factors per image [B, 3].
    
    Args:
        cell: Lattice cell tensor [B, 3, 3] where rows are [a1; a2; a3].
        radius: Cutoff radius.
        pbc: List of boolean flags for periodic boundaries [x, y, z].
        dtype: Computation dtype.
        
    Returns:
        rep: Integer tensor [B, 3] of replication counts along each axis.
    """
    device = cell.device
    batch_size = cell.shape[0]

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    rep = torch.zeros(batch_size, 3, dtype=torch.int32, device=device)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep[:, 0] = (torch.ceil(radius * inv_min_dist_a1) * 2).to(torch.int32)

    if pbc[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep[:, 1] = (torch.ceil(radius * inv_min_dist_a2) * 2).to(torch.int32)

    if pbc[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep[:, 2] = (torch.ceil(radius * inv_min_dist_a3) * 2).to(torch.int32)

    return rep


def sanitize_pbc_flags(data, pbc: Optional[List[bool]] = None) -> List[bool]:
    """
    Validate and extract unified PBC flags from data and user options.
    """
    if pbc is None:
        pbc = [True, True, True]
    else:
        pbc = list(pbc)

    if hasattr(data, "pbc") and data.pbc is not None:
        data_pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data_pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data_pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations."
                )
    return pbc
