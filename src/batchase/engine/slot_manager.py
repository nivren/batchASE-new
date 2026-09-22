"""
Decoupled Dynamic Slot Manager for batched structure optimization.

Maintains active batch slots, tracks per-structure convergence and step count,
and handles dynamic replenishment (slot filling/compaction) to avoid GPU idling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
import ase
from ase.io import read

logger = logging.getLogger("batchase.engine.slot_manager")


class ActiveSlot:
    """Metadata tracking a single crystal structure inside an active batch."""

    __slots__ = ("id", "path", "atoms", "steps", "start_time")

    def __init__(self, id: str, path: str, atoms: ase.Atoms):
        self.id = id
        self.path = path
        self.atoms = atoms
        self.steps: int = 0
        self.start_time: float = time.perf_counter()


class SlotManager:
    """
    Manages the lifecycle and dynamic replenishment of batch slots.
    
    Decoupled from specific potentials and optimizers: provides a clean (keep_indices, num_new_slots)
    contract for stateful optimizers (like BFGS) and stateless optimizers (like FIRE2).
    """

    def __init__(
        self,
        files: List[str],
        batch_size: int,
        max_steps: int = 100,
        max_bnatoms: int = 8000,
        molecule_single: int = 64,
    ) -> None:
        self.files = list(files)
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_bnatoms = max_bnatoms
        self.molecule_single = molecule_single

        self.pending_indices: int = 0
        self.slots: List[ActiveSlot] = []
        self.completed_count: int = 0

    @property
    def total_pending(self) -> int:
        return len(self.files) - self.pending_indices

    @property
    def is_empty(self) -> bool:
        return len(self.slots) == 0 and self.pending_indices >= len(self.files)

    def current_atoms_list(self) -> List[ase.Atoms]:
        return [slot.atoms for slot in self.slots]

    def current_paths(self) -> List[str]:
        return [slot.path for slot in self.slots]

    def fill_initial_batch(self) -> int:
        """Fill batch slots up to batch_size or atomic capacity."""
        cur_natoms = 0
        while len(self.slots) < self.batch_size and self.pending_indices < len(self.files):
            file_path = self.files[self.pending_indices]
            atoms = read(file_path)
            natoms = len(atoms)
            if self.slots and (cur_natoms + natoms > self.max_bnatoms):
                break
            slot_id = Path(file_path).stem
            self.slots.append(ActiveSlot(id=slot_id, path=file_path, atoms=atoms))
            cur_natoms += natoms
            self.pending_indices += 1
        return len(self.slots)

    def update_and_replenish(
        self,
        converge_indices: List[int],
        elapsed_steps: int,
        latest_atoms: List[ase.Atoms],
        energies: List[float],
    ) -> Tuple[List[int], int, List[Dict[str, Any]]]:
        """
        Record completed structures, compact surviving structures, and pop new files into slots.
        
        Args:
            converge_indices: Batch indices that achieved convergence.
            elapsed_steps: Steps taken in the latest optimizer burst.
            latest_atoms: Updated geometry per slot from the optimizer.
            energies: Predicted potential energy per slot.
            
        Returns:
            keep_indices: List of surviving slot indices to retain.
            num_new_slots: Number of newly introduced slots appended to the batch.
            finished_results: List of result records for finished structures.
        """
        now = time.perf_counter()
        finished_results = []
        surviving_slots = []
        keep_indices = []

        # Update step counts and current geometries
        for i, slot in enumerate(self.slots):
            slot.steps += elapsed_steps
            slot.atoms = latest_atoms[i]
            is_converged = i in converge_indices
            is_over_step = slot.steps >= self.max_steps

            if is_converged or is_over_step:
                runtime = now - slot.start_time
                num_mol = len(slot.atoms) / self.molecule_single
                energy_per_mol = (energies[i] / num_mol * 96.485) if num_mol > 0 else energies[i]
                record = {
                    "id": slot.id,
                    "path": slot.path,
                    "steps": slot.steps,
                    "runtime": runtime,
                    "energy": energy_per_mol,
                    "converged": is_converged,
                    "atoms": slot.atoms,
                }
                finished_results.append(record)
                self.completed_count += 1
            else:
                keep_indices.append(i)
                surviving_slots.append(slot)

        # Replenish with new structures
        num_new = 0
        cur_natoms = sum(len(s.atoms) for s in surviving_slots)
        while len(surviving_slots) < self.batch_size and self.pending_indices < len(self.files):
            file_path = self.files[self.pending_indices]
            atoms = read(file_path)
            natoms = len(atoms)
            if surviving_slots and (cur_natoms + natoms > self.max_bnatoms):
                break
            slot_id = Path(file_path).stem
            surviving_slots.append(ActiveSlot(id=slot_id, path=file_path, atoms=atoms))
            cur_natoms += natoms
            self.pending_indices += 1
            num_new += 1

        self.slots = surviving_slots
        return keep_indices, num_new, finished_results
