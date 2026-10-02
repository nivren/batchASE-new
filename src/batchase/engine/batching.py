"""Deterministic structure ordering and complete-batch planning."""

from __future__ import annotations

import os
import random
from pathlib import Path

from ase.io import read

PLANNING_VERSION = 2


def validate_batch_parameters(batch_mode, batch_size, max_batch_atoms, structure_order):
    if batch_mode not in ("bsize", "atoms"):
        raise ValueError("batch_mode must be bsize or atoms")
    if batch_mode == "bsize" and batch_size <= 0:
        raise ValueError("batch_size must be positive in bsize mode")
    if batch_mode == "atoms" and max_batch_atoms <= 0:
        raise ValueError("max_batch_atoms must be positive in atoms mode")
    if structure_order not in ("rand", "syst", "atom"):
        raise ValueError("structure_order must be rand, syst, or atom")


def count_structure_atoms(path: str) -> int:
    """Count the actual atoms after ASE expands CIF symmetry."""
    natoms = len(read(path))
    if natoms <= 0:
        raise ValueError(f"Input structure has no atoms: {path}")
    return natoms


def select_structure_files(files, order="rand", seed=42, num_structures=0):
    """Select without reading CIFs; rand and atom use the same random subset."""
    if order not in ("rand", "syst", "atom"):
        raise ValueError("structure_order must be rand, syst, or atom")
    if not isinstance(num_structures, int) or num_structures < 0:
        raise ValueError("num_structures must be a nonnegative integer")
    ordered = sorted(map(os.fspath, files), key=lambda path: (Path(path).name, path))
    if len({os.path.abspath(path) for path in ordered}) != len(ordered):
        raise ValueError("Batch inputs contain duplicate files")
    count = min(num_structures, len(ordered)) if num_structures else len(ordered)
    if order == "syst":
        return ordered[:count]
    rng = random.Random(seed)
    if count < len(ordered):
        return rng.sample(ordered, count)
    if order == "rand":
        rng.shuffle(ordered)
    return ordered


def _order_selected_files(files, order, atom_counts):
    if order == "atom":
        return sorted(files, key=lambda path: (-atom_counts[path], Path(path).name, path))
    return list(files)


def order_structure_files(files, order="rand", seed=42, atom_counts=None, num_structures=0):
    """Select first, then sort only the sampled atom-mode structures."""
    selected = select_structure_files(files, order, seed, num_structures)
    counts = atom_counts
    if order == "atom" and counts is None:
        counts = {path: count_structure_atoms(path) for path in selected}
    return _order_selected_files(selected, order, counts)


def describe_batch(files, atom_counts, batch_id, atom_budget=0):
    """Record batch composition, including nonlinear size indicators."""
    counts = [atom_counts[path] for path in files]
    total = sum(counts)
    return {
        "batch_id": batch_id,
        "files": list(files),
        "nfiles": len(files),
        "natoms": total,
        "min_structure_atoms": min(counts),
        "max_structure_atoms": max(counts),
        "sum_squared_atoms": sum(n * n for n in counts),
        "atom_budget_fill_ratio": total / atom_budget if atom_budget else None,
    }


def build_batch_plan(files, batch_mode="bsize", batch_size=4, max_batch_atoms=0,
                     structure_order="rand", structure_order_seed=42, num_structures=0):
    """Plan all inputs without assigning batches to particular workers.

    bsize: use ceil(N / batch_size) batches, balanced to within one structure.
    atoms: first-fit packing in the selected order, constrained only by atoms.
    """
    validate_batch_parameters(batch_mode, batch_size, max_batch_atoms, structure_order)
    paths = list(map(os.fspath, files))
    selected = select_structure_files(paths, structure_order, structure_order_seed, num_structures)
    plan = _build_selected_batch_plan(selected, batch_mode, batch_size, max_batch_atoms,
                                      structure_order, structure_order_seed)
    plan.update(num_candidates=len(paths), num_structures=num_structures)
    return plan


def _build_selected_batch_plan(paths, batch_mode, batch_size, max_batch_atoms,
                               structure_order, structure_order_seed):
    """Pack an already-selected sequence without sampling or shuffling again."""
    atom_counts = {path: count_structure_atoms(path) for path in paths}
    ordered = _order_selected_files(paths, structure_order, atom_counts)
    groups = []
    if batch_mode == "bsize" and ordered:
        num_batches = (len(ordered) + batch_size - 1) // batch_size
        size, remainder = divmod(len(ordered), num_batches)
        offset = 0
        for index in range(num_batches):
            count = size + (index < remainder)
            groups.append(ordered[offset:offset + count])
            offset += count
    elif batch_mode == "atoms":
        totals = []
        for path in ordered:
            natoms = atom_counts[path]
            if natoms > max_batch_atoms:
                raise ValueError(
                    f"Structure exceeds atom budget: {path} "
                    f"({natoms} > {max_batch_atoms}); increase max_batch_atoms"
                )
            for index, total in enumerate(totals):
                if total + natoms <= max_batch_atoms:
                    groups[index].append(path)
                    totals[index] += natoms
                    break
            else:
                groups.append([path])
                totals.append(natoms)
    budget = max_batch_atoms if batch_mode == "atoms" else 0
    return {
        "mode": "shared-queue",
        "planning_version": PLANNING_VERSION,
        "packing_algorithm": "balanced-count" if batch_mode == "bsize" else "first-fit",
        "batch_mode": batch_mode,
        "structure_order": structure_order,
        "structure_order_seed": structure_order_seed,
        "batch_size_limit": batch_size if batch_mode == "bsize" else None,
        "max_batch_atoms": budget,
        "num_files": len(paths),
        "num_batches": len(groups),
        "ordered_files": ordered,
        "batches": [describe_batch(group, atom_counts, index, budget)
                    for index, group in enumerate(groups)],
    }
