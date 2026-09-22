"""
Common utility functions for batchASE: CIF atom parsing, batch collation, and I/O helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from itertools import product
from typing import List, Union

import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData


def count_atoms_cif(file_path: Union[str, Path]) -> int:
    """
    Count the number of atomic sites in a CIF file without parsing full geometry.
    
    Fast streaming parse using _atom_site_ loop detection.
    """
    in_atom_site = False
    natoms = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        while line := f.readline():
            line_lower = line.lower().strip()
            if line_lower.startswith("loop_"):
                in_atom_site = False
                continue
            if "_atom_site_" in line_lower:
                in_atom_site = True
                continue
            if in_atom_site:
                if line_lower.startswith("_"):
                    in_atom_site = False
                    continue
                elif line.strip():
                    natoms += 1
    return natoms


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating parent directories if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def collate(data_list: List[BaseData]):
    """Collate PyG data objects with slice indexing."""
    keys = data_list[0].keys()
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        val = item[key]
        data[key].append(val)
        if torch.is_tensor(val):
            s = slices[key][-1] + val.size(item.__cat_dim__(key, val))
        elif isinstance(val, (int, float)):
            s = slices[key][-1] + 1
        else:
            raise ValueError(f"Unsupported attribute type for key {key}: {type(val)}")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = [item.num_nodes for item in data_list]

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(data[key], dim=data.__cat_dim__(key, data_list[0][key]))
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


def data_list_collater(
    data_list: List[BaseData],
    otf_graph: bool = True,
    to_dict: bool = False,
) -> Union[BaseData, dict[str, torch.Tensor]]:
    """
    Collate a list of PyG Data objects into a unified Batch.
    
    Default otf_graph=True: On-the-fly graph construction where edges are computed
    dynamically by pbc_graph kernels at each step, avoiding static edge collation overhead.
    """
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = [data.edge_index[1, :].shape[0] for data in data_list]
            batch.neighbors = torch.tensor(n_neighbors)
        except Exception:
            pass

    if to_dict:
        batch = dict(batch.items())

    return batch
