"""
Pure PyTorch CPU fallback implementation for periodic boundary condition (PBC) radius graph.
"""

from __future__ import annotations

import torch
from typing import Optional, List


def radius_graph_pbc_cpu(
    data,
    radius: float,
    pbc: Optional[List[bool]] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute periodic neighbor graph using pure PyTorch operations (CPU / fallback).
    
    Args:
        data: Data batch containing pos, natoms, cell, and optional pbc.
        radius: Cutoff distance.
        pbc: Periodic boundary conditions per axis [x, y, z]. Default [True, True, True].
        dtype: Computation dtype.
        
    Returns:
        edge_index: [2, E] tensor of (src, dst) atom indices.
        unit_cell: [E, 3] cell replication shift vector for each edge.
        num_neighbors_image: [B] number of edges per image.
    """
    if pbc is None:
        pbc = [True, True, True]

    device = data.pos.device
    batch_size = len(data.natoms)

    atom_pos = data.pos.to(dtype)
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # Index offset for cumulative atoms per image
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )
    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand

    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    max_rep = [int(2 * rep_a1.max().item()), int(2 * rep_a2.max().item()), int(2 * rep_a3.max().item())]

    # Expand image cell
    cell_expand = torch.repeat_interleave(data.cell.to(dtype), num_atoms_per_image_sqr, dim=0)

    # Unit cell offsets
    u0 = torch.arange(-max_rep[0], max_rep[0] + 1, device=device) if pbc[0] else torch.zeros(1, device=device)
    u1 = torch.arange(-max_rep[1], max_rep[1] + 1, device=device) if pbc[1] else torch.zeros(1, device=device)
    u2 = torch.arange(-max_rep[2], max_rep[2] + 1, device=device) if pbc[2] else torch.zeros(1, device=device)

    u0, u1, u2 = torch.meshgrid(u0, u1, u2, indexing="ij")
    unit_cell_shifts = torch.stack([u0.flatten(), u1.flatten(), u2.flatten()], dim=-1).to(dtype)

    all_edge_index = []
    all_cell_offsets = []

    # Iterate over periodic shifts
    for shift in unit_cell_shifts:
        # Shift in Cartesian coordinates: shift * cell
        cart_shift = torch.matmul(shift.view(1, 3), cell_expand).squeeze(1)
        shifted_pos2 = pos2 + cart_shift
        dist = torch.norm(pos1 - shifted_pos2, p=2, dim=-1)

        # Filter self-loops at offset (0, 0, 0)
        is_zero_shift = (shift == 0).all()
        mask = dist <= radius
        if is_zero_shift:
            mask = mask & (index1 != index2)

        if mask.any():
            all_edge_index.append(torch.stack([index2[mask], index1[mask]], dim=0))
            all_cell_offsets.append(shift.unsqueeze(0).expand(mask.sum(), -1))

    if len(all_edge_index) > 0:
        edge_index = torch.cat(all_edge_index, dim=1)
        unit_cell = torch.cat(all_cell_offsets, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        unit_cell = torch.zeros((0, 3), dtype=dtype, device=device)

    # Count edges per image
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_atoms_per_image
    )
    dst_batch = batch_idx[edge_index[1]] if edge_index.shape[1] > 0 else torch.zeros(0, dtype=torch.long, device=device)
    num_neighbors_image = torch.bincount(dst_batch, minlength=batch_size)

    return edge_index, unit_cell, num_neighbors_image
