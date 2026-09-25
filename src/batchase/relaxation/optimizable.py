"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Code based on ase.optimize
"""

from __future__ import annotations

from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Any, Generator
import numpy as np
import torch
import logging
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import voigt_6_to_full_3x3_stress


from .ase_utils import batch_to_atoms
from itertools import product

from ..potentials.base import BatchPotential

try:
    from torch_scatter import scatter
except ImportError:
    scatter = None

# Define dummy classes for when imports fail
class _DummyCalculator:
    pass


try:
    from mace.calculators import MACECalculator
except ImportError:
    logging.debug("Unable to import MACECalculator.")
    MACECalculator = _DummyCalculator

try:
    from chgnet.model.dynamics import CHGNetCalculator
except ImportError:
    logging.debug("Unable to import CHGNetCalculator.")
    CHGNetCalculator = _DummyCalculator

try:
    from sevenn.calculator import (
        SevenNetCalculator,
        SevenNetD3Calculator,
        D3Calculator,
    )
except ImportError:
    logging.debug("Unable to import SevenNetCalculator.")
    SevenNetCalculator = _DummyCalculator
    SevenNetD3Calculator = _DummyCalculator
    D3Calculator = _DummyCalculator

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    logging.debug("Unable to import FAIRChemCalculator.")
    FAIRChemCalculator = _DummyCalculator


# this can be removed after pinning ASE dependency >= 3.23
try:
    from ase.optimize.optimize import Optimizable
except ImportError:

    class Optimizable:
        pass


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from numpy.typing import NDArray
    from torch_geometric.data import Batch


ALL_CHANGES: set[str] = {
    "pos",
    "atomic_numbers",
    "cell",
    "pbc",
}


# @torch.compile
def compare_batches(
    batch1: Batch | None,
    batch2: Batch,
    tol: float = 1e-6,
    excluded_properties: set[str] | None = None,
) -> list[str]:
    """Compare properties between two batches

    Args:
        batch1: atoms batch
        batch2: atoms batch
        tol: tolerance used to compare equility of floating point properties
        excluded_properties: list of properties to exclude from comparison

    Returns:
        list of system changes, property names that are differente between batch1 and batch2
    """
    system_changes = []

    if batch1 is None:
        system_changes = ALL_CHANGES
    else:
        properties_to_check = set(ALL_CHANGES)
        if excluded_properties:
            properties_to_check -= set(excluded_properties)

        # Check properties that aren't
        for prop in ALL_CHANGES:
            if prop in properties_to_check:
                properties_to_check.remove(prop)
                if not torch.allclose(
                    getattr(batch1, prop), getattr(batch2, prop), atol=tol
                ):
                    system_changes.append(prop)

    return system_changes


class OptimizableBatch(Optimizable):
    """A Batch version of ase Optimizable Atoms

    This class can be used with ML relaxations in fairchem.core.relaxations.ml_relaxation
    or in ase relaxations classes, i.e. ase.optimize.lbfgs
    """

    ignored_changes: ClassVar[set[str]] = set()

    def __init__(
        self,
        batch: Batch,
        trainer: Any = None,  # 兼容旧接口：MACECalculator | CHGNetCalculator | SevenNetCalculator | FAIRChemCalculator
        backend: BatchPotential | None = None,  # v0.5.0 起：统一适配层，MACE 走此路径
        transform: torch.nn.Module | None = None,
        mask_converged: bool = True,
        numpy: bool = False,
        masked_eps: float = 1e-8,
        compute_stress: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize Optimizable Batch

        Args:
            batch: A batch of atoms graph data
            trainer: 旧接口，An instance of a BaseTrainer derived class（无 backend 时使用）
            backend: v0.5.0 起推荐的 BatchPotential 适配器（MACE 批处理路径）
            transform: graph transform
            mask_converged: if true will mask systems in batch that are already converged
            numpy: whether to cast results to numpy arrays
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
            compute_stress: whether to compute stress during prediction
            dtype: data type for tensor operations (torch.float32 or torch.float64)
        """
        if trainer is None and backend is None:
            raise ValueError("OptimizableBatch 必须提供 trainer 或 backend 之一")
        self.backend = backend
        self.trainer = trainer
        self._calc_device = (
            backend.device if backend is not None else trainer.device
        )
        self.batch = batch.to(self._calc_device)
        self.transform = transform
        self.numpy = numpy
        self.mask_converged = mask_converged
        self._cached_batch = None
        self._update_mask = None
        self.torch_results = {}
        self.results = {}
        self._eps = masked_eps
        self.dtype = dtype

        self.otf_graph = True  # trainer._unwrapped_model.otf_graph
        if not self.otf_graph and "edge_index" not in self.batch:
            self.update_graph()

        self.batch.pos = self.batch.pos.to(dtype=self.dtype)
        self.batch.cell = self.batch.cell.to(dtype=self.dtype)
        self.compute_stress = compute_stress

        # Determine backend/calculator type once during initialization
        self._calculator_type = (
            backend.kind if backend is not None else self._determine_calculator_type()
        )
        logging.debug(
            f"OptimizableBatch initialized with calculator type: {self._calculator_type}"
        )

    def _determine_calculator_type(self) -> str:
        """Determine the type of calculator to avoid repeated isinstance checks."""
        # Check against actual imported classes, not dummy classes
        trainer_class_name = type(self.trainer).__name__
        trainer_module = type(self.trainer).__module__

        if (
            "mace" in trainer_module.lower()
            or trainer_class_name == "MACECalculator"
        ):
            return "mace"
        elif (
            "chgnet" in trainer_module.lower()
            or trainer_class_name == "CHGNetCalculator"
        ):
            return "chgnet"
        elif "sevenn" in trainer_module.lower() or trainer_class_name in [
            "SevenNetCalculator",
            "SevenNetD3Calculator",
            "D3Calculator",
        ]:
            return "sevennet"
        elif (
            "fairchem" in trainer_module.lower()
            or trainer_class_name == "FAIRChemCalculator"
        ):
            return "fairchem"
        else:
            return "default"

    @property
    def device(self):
        return self._calc_device

    @property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch."""
        return self.batch.batch

    @property
    def converged_mask(self):
        if self._update_mask is not None:
            return torch.logical_not(self._update_mask)
        return None

    @property
    def update_mask(self):
        if self._update_mask is None:
            return torch.ones(len(self.batch), dtype=bool)
        return self._update_mask

    @property
    def converge_indices_list(self):
        return torch.where(~self.update_mask)[0].tolist()

    @property
    def elem_per_group(self):
        # This return value actually represents the number of elements
        # in a group within a batch. Each group corresponds to batch_indices.
        # It will count the number of CELL elements in each group.
        return torch.bincount(self.batch_indices)

    @property
    def batch_size(self):
        return len(torch.unique(self.batch_indices))

    def check_state(self, batch: Batch, tol: float = 1e-12) -> list[str]:
        """Check for any system changes since last calculation."""
        return compare_batches(
            self._cached_batch,
            batch,
            tol=tol,
            excluded_properties=set(self.ignored_changes),
        )

    def _predict(self) -> None:
        """Run prediction if batch has any changes."""
        system_changes = self.check_state(self.batch)
        if len(system_changes) > 0:
            if self.backend is not None:
                # v0.5.0 统一适配层路径：backend 自行构图并调 model.forward(dict)
                self.torch_results = self.backend.predict(
                    self.batch, compute_stress=self.compute_stress
                )
                self.batch.pos = self.batch.pos.to(self.dtype)
                self.batch.cell = self.batch.cell.to(self.dtype)
            elif self._calculator_type == "mace":
                raise RuntimeError(
                    "MACE 后端必须通过 backend= 传入（统一走 batchase.potentials 适配层）。"
                )
            elif self._calculator_type == "fairchem":
                # TODO: FAIRChemCalculator does not support batch prediction yet
                atoms_list = batch_to_atoms(
                    self.batch, results=None, wrap_pos=False, eps=1e-17
                )
                self.torch_results = self.trainer.predict(atoms_list=atoms_list)
            elif self._calculator_type == "chgnet":
                atoms_list = batch_to_atoms(
                    self.batch, results=None, wrap_pos=False, eps=1e-17
                )
                model_prediction = self.trainer.predict(
                    atoms_list=atoms_list, task="efs"
                )
                results = {
                    "energy": torch.tensor(
                        [pred["e"].item() for pred in model_prediction],
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "forces": torch.vstack(
                        [
                            torch.from_numpy(pred["f"]).to(
                                device=self.device, dtype=self.dtype
                            )
                            for pred in model_prediction
                        ]
                    ),
                    "stress": torch.vstack(
                        [
                            torch.from_numpy(pred["s"]).to(
                                device=self.device, dtype=self.dtype
                            )
                            for pred in model_prediction
                        ]
                    ).view(-1, 3, 3),
                }
                self.torch_results = results
            elif self._calculator_type == "sevennet":
                atoms_list = batch_to_atoms(
                    self.batch, results=None, wrap_pos=False, eps=1e-17
                )
                self.torch_results = self.trainer.predict(atoms_list=atoms_list)
            else:  # default case
                self.torch_results = self.trainer.predict(
                    self.batch, per_image=False, disable_tqdm=True
                )
            # save only subset of props in simple namespace instead of cloning the whole batch to save memory
            changes = ALL_CHANGES - set(self.ignored_changes)
            self._cached_batch = SimpleNamespace(
                **{prop: self.batch[prop].clone() for prop in changes}
            )

    def get_property(
        self, name, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get a predicted property by name."""
        self._predict()
        if self.numpy:
            self.results = {
                key: pred.item() if pred.numel() == 1 else pred.cpu().numpy()
                for key, pred in self.torch_results.items()
            }
        else:
            self.results = self.torch_results

        if name not in self.results:
            raise PropertyNotImplementedError(
                f"{name} not present in this calculation"
            )

        return (
            self.results[name]
            if no_numpy is False
            else self.torch_results[name]
        )

    def get_positions(self) -> torch.Tensor | NDArray:
        """Get the batch positions"""
        pos = self.batch.pos.clone()
        if self.numpy:
            if self.mask_converged:
                pos[~self.update_mask[self.batch.batch]] = self._eps
            pos = pos.cpu().numpy()

        return pos

    def set_positions(self, positions: torch.Tensor | NDArray) -> None:
        """Set the atom positions in the batch."""
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(
                positions, dtype=self.dtype, device=self.device
            )
        else:
            positions = positions.to(dtype=self.dtype, device=self.device)

        if self.mask_converged and self._update_mask is not None:
            mask = self.update_mask[self.batch.batch]
            self.batch.pos[mask] = positions[mask]
        else:
            self.batch.pos = positions

        if not self.otf_graph:
            self.update_graph()

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get predicted batch forces."""
        forces = self.get_property("forces", no_numpy=no_numpy)
        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            if isinstance(forces, np.ndarray):
                fixed_idx = fixed_idx.tolist()
            forces[fixed_idx] = 0.0
        return forces.view(-1, 3)

    def get_potential_energy(self, **kwargs) -> torch.Tensor | NDArray:
        """Get predicted energy as the sum of all batch energies."""
        # ASE 3.22.1 expects a check for force_consistent calculations
        if kwargs.get("force_consistent", False) is True:
            raise PropertyNotImplementedError(
                "force_consistent calculations are not implemented"
            )
        if (
            len(self.batch) == 1
        ):  # unfortunately batch size 1 returns a float, not a tensor
            return self.get_property("energy")
        return self.get_property("energy").sum()

    def get_potential_energies(self) -> torch.Tensor | NDArray:
        """Get the predicted energy for each system in batch."""
        energies = self.get_property("energy")
        if hasattr(energies, "view"):
            return energies.view(-1)
        return energies

    def get_cells(self) -> torch.Tensor:
        """Get batch crystallographic cells."""
        return self.batch.cell

    def set_cells(
        self, cells: torch.Tensor | NDArray, scale_atoms=False
    ) -> None:
        """Set batch cells."""
        assert self.batch.cell.shape == cells.shape, "Cell shape mismatch"
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=self.dtype, device=self.device)
        cells = cells.to(dtype=self.dtype, device=self.device)
        if scale_atoms:
            from ase.geometry.cell import complete_cell

            # M = torch.linalg.solve(
            #     self.batch.cell.view(-1, 3, 3),
            #     cells.view(-1, 3, 3),
            # )
            # TODO: need to implement a sparse version.
            # tmp_pos = torch.matmul(self.batch.pos, M.reshape(-1,3))
            for i in range(self.batch_size):
                if not self.update_mask[i]:
                    continue
                M = np.linalg.solve(
                    complete_cell(self.batch.cell[i].cpu().detach().numpy()),
                    complete_cell(cells[i].cpu().detach().numpy()),
                )
                pos_update_mask = self.batch.batch == i
                self.batch.pos[pos_update_mask] = torch.matmul(
                    self.batch.pos[pos_update_mask],
                    torch.from_numpy(M).to(self.device).reshape(-1, 3),
                )
        self.batch.cell[self.update_mask] = cells[self.update_mask]

    def get_volumes(self) -> torch.Tensor:
        """Get a tensor of volumes for each cell in batch"""
        cells = self.get_cells()
        return torch.linalg.det(cells)

    def iterimages(self) -> Generator[Batch, None, None]:
        # XXX document purpose of iterimages - this is just needed to work with ASE optimizers
        yield self.batch

    def get_max_forces(
        self, forces: torch.Tensor | None = None, apply_constraint: bool = False
    ) -> torch.Tensor:
        """Get the maximum forces per structure in batch"""
        if forces is None:
            forces = self.get_forces(
                apply_constraint=apply_constraint, no_numpy=True
            )
        if scatter is not None:
            return scatter(
                (forces**2).sum(axis=1).sqrt(), self.batch_indices, reduce="max"
            )
        else:
            index = self.batch_indices
            src = (forces.pow(2).sum(dim=1)).sqrt()
            num_groups = int(index.max().item()) + 1
            out = torch.full(
                (num_groups,),
                float('-inf'),
                device=src.device,
                dtype=src.dtype
            )
            result = out.scatter_reduce(
                dim=0,
                index=index,
                src=src,
                reduce="amax",
                include_self=True
            )
            return result

    def converged(
        self,
        forces: torch.Tensor | NDArray | None,
        fmax: float,
        max_forces: torch.Tensor | None = None,
        f_upper_limit: float = 1e20,
    ) -> bool:
        """Check if norm of all predicted forces are below fmax"""
        if forces is not None:
            if isinstance(forces, np.ndarray):
                forces = torch.tensor(
                    forces, device=self.device, dtype=self.dtype
                )
            max_forces = self.get_max_forces(forces)
        elif max_forces is None:
            max_forces = self.get_max_forces()

        # Update mask is True for forces that are greater than fmax AND less than f_upper_limit
        update_mask = torch.logical_and(
            max_forces.ge(fmax), max_forces.le(f_upper_limit)
        )
        # update cached mask
        if self.mask_converged:
            if self._update_mask is None:
                self._update_mask = update_mask
            else:
                # some models can have random noise in their predictions, so the mask is updated by
                # keeping all previously converged structures masked even if new force predictions
                # push it slightly above threshold
                self._update_mask = torch.logical_and(
                    self._update_mask, update_mask
                )
            update_mask = self._update_mask

        return not torch.any(update_mask).item()

    def get_atoms_list(self) -> list[Atoms]:
        """Get ase Atoms objects corresponding to the batch"""
        self._predict()  # in case no predictions have been run
        return batch_to_atoms(self.batch, results=self.torch_results)

    def update_graph(self):
        """Update the graph if model does not use otf_graph."""
        graph = self.trainer._unwrapped_model.generate_graph(self.batch)
        self.batch.edge_index = graph.edge_index
        self.batch.cell_offsets = graph.cell_offsets
        self.batch.neighbors = graph.neighbors
        if self.transform is not None:
            self.batch = self.transform(self.batch)

    def __len__(self) -> int:
        # TODO: this might be changed in ASE to be 3 * len(self.atoms)
        return len(self.batch.pos)


class OptimizableUnitCellBatch(OptimizableBatch):
    """Modify the supercell and the atom positions in relaxations.

    Based on ase UnitCellFilter to work on data batches
    """

    def __init__(
        self,
        batch: Batch,
        trainer: Any = None,  # 兼容旧接口
        backend: BatchPotential | None = None,  # v0.5.0 推荐
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
        mask_converged: bool = True,
        mask: Sequence[bool] | None = None,
        cell_factor: float | torch.Tensor | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
        masked_eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
    ):
        """Create a filter that returns the forces and unit cell stresses together, for simultaneous optimization.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        Args:
            batch: A batch of atoms graph data
            trainer: 旧接口，An instance of a BaseTrainer derived class
            backend: v0.5.0 推荐的 BatchPotential 适配器
            transform: graph transform
            numpy: whether to cast results to numpy arrays
            mask_converged: if true will mask systems in batch that are already converged
            mask: a boolean mask specifying which strain components are allowed to relax
            cell_factor:
                Factor by which deformation gradient is multiplied to put
                it on the same scale as the positions when assembling
                the combined position/cell vector. The stress contribution to
                the forces is scaled down by the same factor. This can be thought
                of as a very simple preconditioner. Default is number of atoms
                which gives approximately the correct scaling.
            hydrostatic_strain:
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by np.diag([np.trace(virial)]*3).
            constant_volume:
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve. Note: this only approximately conserves
                the volume and breaks energy/force consistency so can only be
                used with optimizers that do require a line minimisation
                (e.g. FIRE).
            scalar_pressure:
                Applied external hydrostatic pressure in eV/A^3 for enthalpy pV term
                (e.g., 0.0006 eV/A^3 ≈ 0.096 GPa / ~1000 bar). As above, this
                breaks energy/force consistency.
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
            dtype: data type for tensor operations (torch.float32 or torch.float64)
        """
        super().__init__(
            batch=batch,
            trainer=trainer,
            backend=backend,
            transform=transform,
            numpy=numpy,
            mask_converged=mask_converged,
            masked_eps=masked_eps,
            compute_stress=True,
            dtype=dtype,
        )

        self.orig_cells = self.get_cells().clone()
        self.stress = None

        if mask is None:
            # mask = torch.eye(3, device=self.device)
            mask = torch.ones(6, device=self.device)

        # TODO make sure mask is on GPU
        if mask.shape == (6,):
            self.mask = torch.tensor(
                voigt_6_to_full_3x3_stress(mask.detach().cpu()),
                device=self.device,
            )
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError("shape of mask should be (3,3) or (6,)")

        if isinstance(cell_factor, float):
            cell_factor = cell_factor * torch.ones(
                (3 * len(batch), 1), requires_grad=False
            )
        if cell_factor is None:
            cell_factor = self.batch.natoms.repeat_interleave(3).unsqueeze(
                dim=1
            )

        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.pressure = scalar_pressure * torch.eye(3, device=self.device)
        self.cell_factor = cell_factor
        self.stress = None
        self._batch_trace = torch.vmap(torch.trace)
        self._batch_diag = torch.vmap(
            lambda x: x * torch.eye(3, device=x.device)
        )

    @cached_property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch.

        We augment this to specify the batch indices for augmented positions and forces.
        """
        augmented_batch = torch.repeat_interleave(
            torch.arange(
                len(self.batch),
                dtype=self.batch.batch.dtype,
                device=self.device,
            ),
            3,
        )
        return torch.cat([self.batch.batch, augmented_batch])

    def deform_grad(self):
        """Get the cell deformation matrix"""
        return torch.transpose(
            torch.linalg.solve(self.orig_cells, self.get_cells()), 1, 2
        )

    def get_positions(self):
        """Get positions and cell deformation gradient."""
        cur_deform_grad = self.deform_grad()
        natoms = self.batch.num_nodes
        pos = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            dtype=self.batch.pos.dtype,
            device=self.device,
        )

        # Augmented positions are the self.atoms.positions but without the applied deformation gradient
        pos[:natoms] = torch.linalg.solve(
            cur_deform_grad[self.batch.batch, :, :],
            self.batch.pos.view(-1, 3, 1),
        ).view(-1, 3)
        # cell DOFs are the deformation gradient times a scaling factor
        pos[natoms:] = self.cell_factor * cur_deform_grad.view(-1, 3)
        return pos.cpu().numpy() if self.numpy else pos

    def set_positions(self, positions: torch.Tensor | NDArray) -> None:
        """Set positions and cell.

        positions has shape (natoms + ncells * 3, 3).
        the first natoms rows are the positions of the atoms, the last nsystems * three rows are the deformation tensor
        for each cell.
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(
                positions, dtype=self.dtype, device=self.device
            )
        else:
            positions = positions.to(dtype=self.dtype, device=self.device)

        natoms = self.batch.num_nodes
        new_atom_positions = positions[:natoms]
        new_deform_grad = (positions[natoms:] / self.cell_factor).view(-1, 3, 3)

        # TODO check that in fact symmetry is preserved setting cells and positions
        # Set the new cell from the original cell and the new deformation gradient.  Both current and final structures
        # should preserve symmetry.
        new_cells = torch.bmm(
            self.orig_cells, torch.transpose(new_deform_grad, 1, 2)
        )
        self.set_cells(new_cells)

        # Set the positions from the ones passed in (which are without the deformation gradient applied) and the new
        # deformation gradient. This should also preserve symmetry
        new_atom_positions = torch.bmm(
            new_atom_positions.view(-1, 1, 3),
            torch.transpose(
                new_deform_grad[self.batch.batch, :, :].view(-1, 3, 3), 1, 2
            ),
        )
        super().set_positions(new_atom_positions.view(-1, 3))

    def get_potential_energy(self, **kwargs):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = super().get_potential_energy(**kwargs)
        return atoms_energy + self.pressure[0, 0] * self.get_volumes().sum()

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get forces and unit cell stress."""
        stress = self.get_property("stress", no_numpy=True).view(-1, 3, 3)
        atom_forces = self.get_property("forces", no_numpy=True)

        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            atom_forces[fixed_idx] = 0.0

        volumes = self.get_volumes().view(-1, 1, 1)
        # virial = -volumes * stress + self.pressure.view(-1, 3, 3)
        virial = -volumes * (stress + self.pressure.view(-1, 3, 3))
        # print(f'&&& virial0: {virial}')
        cur_deform_grad = self.deform_grad()
        atom_forces = torch.bmm(
            atom_forces.view(-1, 1, 3),
            cur_deform_grad[self.batch.batch, :, :].view(-1, 3, 3),
        )
        virial = torch.linalg.solve(
            cur_deform_grad, torch.transpose(virial, dim0=1, dim1=2)
        )
        virial = torch.transpose(virial, dim0=1, dim1=2)

        # print(f'&&& virial1: {virial}')

        # TODO this does not work yet! maybe _batch_trace gives an issue
        if self.hydrostatic_strain:
            virial = self._batch_diag(self._batch_trace(virial) / 3.0)

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask.view(-1, 3, 3)

        if self.constant_volume:
            virial[:, range(3), range(3)] -= (
                self._batch_trace(virial).view(3, -1) / 3.0
            )

        natoms = self.batch.num_nodes
        augmented_forces = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            device=self.device,
            dtype=atom_forces.dtype,
        )
        # print(f'&&& atom_forces: {atom_forces}')
        # print(f'&&& virial2: {virial}')
        augmented_forces[:natoms] = atom_forces.view(-1, 3)
        augmented_forces[natoms:] = virial.view(-1, 3) / self.cell_factor

        self.stress = -virial.view(-1, 9) / volumes.view(-1, 1)

        if self.numpy and not no_numpy:
            augmented_forces = augmented_forces.cpu().numpy()

        # print(f'&&& augmented_forces: {augmented_forces}')

        return augmented_forces

    def __len__(self):
        return len(self.batch.pos) + 3 * len(self.batch)

    def get_potential_energies(self) -> torch.Tensor:
        """Get the predicted energy for each system in batch."""
        return (
            self.get_property("energy").view(-1)
            + self.pressure[0, 0] * self.get_volumes()
        )


class OptimizableFrechetCellBatch(OptimizableBatch):
    """Modify the supercell and the atom positions in relaxations using Fréchet derivatives.

    Based on ase FrechetCellFilter to work on data batches, providing better convergence
    for cell variables compared to OptimizableUnitCellBatch.
    """

    def __init__(
        self,
        batch: Batch,
        trainer: Any = None,  # 兼容旧接口
        backend: BatchPotential | None = None,  # v0.5.0 推荐
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
        mask_converged: bool = True,
        mask: Sequence[bool] | None = None,
        exp_cell_factor: float | torch.Tensor | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
        masked_eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
    ):
        """Create a filter that returns the forces and unit cell stresses together, for simultaneous optimization.

        Uses Fréchet derivatives for better convergence with respect to cell variables.

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            numpy: whether to cast results to numpy arrays
            mask_converged: if true will mask systems in batch that are already converged
            mask: a boolean mask specifying which strain components are allowed to relax
            exp_cell_factor:
                Scaling factor for cell variables. The cell gradients in get_forces() 
                is divided by exp_cell_factor. By default, set to number of atoms.
                We recommend to set an extensive value for this parameter.
            hydrostatic_strain:
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by torch.diag([torch.trace(virial)]*3).
            constant_volume:
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve.
            scalar_pressure:
                Applied external hydrostatic pressure in eV/A^3 for enthalpy pV term
                (e.g., 0.0006 eV/A^3 ≈ 0.096 GPa / ~1000 bar). As above, this
                breaks energy/force consistency.
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
            dtype: data type for tensor operations (torch.float32 or torch.float64)
        """
        super().__init__(
            batch=batch,
            trainer=trainer,
            backend=backend,
            transform=transform,
            numpy=numpy,
            mask_converged=mask_converged,
            masked_eps=masked_eps,
            compute_stress=True,
            dtype=dtype,
        )

        self.orig_cells = self.get_cells().clone()
        self.stress = None

        if mask is None:
            mask = torch.ones(6, device=self.device)

        # TODO make sure mask is on GPU
        if mask.shape == (6,):
            self.mask = torch.tensor(
                voigt_6_to_full_3x3_stress(mask.detach().cpu()),
                device=self.device,
            )
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError("shape of mask should be (3,3) or (6,)")

        if isinstance(exp_cell_factor, float):
            exp_cell_factor = exp_cell_factor * torch.ones(
                (3 * len(batch), 1), requires_grad=False
            )
        if exp_cell_factor is None:
            exp_cell_factor = self.batch.natoms.repeat_interleave(3).unsqueeze(
                dim=1
            )

        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.pressure = scalar_pressure * torch.eye(3, device=self.device)
        self.exp_cell_factor = exp_cell_factor
        self.stress = None
        self._batch_trace = torch.vmap(torch.trace)
        self._batch_diag = torch.vmap(
            lambda x: x * torch.eye(3, device=x.device)
        )

        # Import scipy functions for matrix logarithm and exponential
        try:
            from scipy.linalg import expm, expm_frechet, logm
            self.expm = expm
            self.logm = logm
            self.expm_frechet = expm_frechet
        except ImportError:
            raise ImportError("scipy is required for FrechetCellFilter functionality")

    @cached_property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch.

        We augment this to specify the batch indices for augmented positions and forces.
        """
        augmented_batch = torch.repeat_interleave(
            torch.arange(
                len(self.batch),
                dtype=self.batch.batch.dtype,
                device=self.device,
            ),
            3,
        )
        return torch.cat([self.batch.batch, augmented_batch])

    def deform_grad(self):
        """Get the cell deformation matrix"""
        return torch.transpose(
            torch.linalg.solve(self.orig_cells, self.get_cells()), 1, 2
        )

    def get_positions(self):
        """Get positions and cell deformation gradient (log form)."""
        cur_deform_grad = self.deform_grad()
        natoms = self.batch.num_nodes
        pos = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            dtype=self.batch.pos.dtype,
            device=self.device,
        )

        # Augmented positions are the self.atoms.positions but without the applied deformation gradient
        pos[:natoms] = torch.linalg.solve(
            cur_deform_grad[self.batch.batch, :, :],
            self.batch.pos.view(-1, 3, 1),
        ).view(-1, 3)
        
        # cell DOFs are the log of deformation gradient times a scaling factor
        # Convert to numpy for scipy operations, then back to torch
        cur_deform_grad_np = cur_deform_grad.cpu().detach().numpy()
        for i in range(len(self.get_cells())):
            # Follow ASE's FrechetCellFilter logic:
            # 1. First get UnitCellFilter-style positions (cell_factor * deform_grad)
            # 2. Then apply logm and multiply by exp_cell_factor
            
            # Step 1: Mimic UnitCellFilter's cell DOFs (equivalent to cell_factor * cur_deform_grad)
            # Since we don't have cell_factor in this class, we use the number of atoms as in UnitCellFilter
            natoms_system = self.batch.natoms[i].item()
            ucf_cell_dofs = natoms_system * cur_deform_grad_np[i]
            
            # Step 2: Apply logm to the UnitCellFilter-style DOFs, then scale
            log_ucf_cell_dofs = self.logm(ucf_cell_dofs)
            scaling_factor = self.exp_cell_factor[i*3].item()
            
            result = torch.from_numpy(log_ucf_cell_dofs).to(self.device) * scaling_factor
            pos[natoms + i*3:natoms + (i+1)*3] = result
        
        return pos.cpu().numpy() if self.numpy else pos

    def set_positions(self, positions: torch.Tensor | NDArray) -> None:
        """Set positions and cell.

        positions has shape (natoms + ncells * 3, 3).
        the first natoms rows are the positions of the atoms, the last nsystems * three rows are the log deformation tensor
        for each cell.
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(
                positions, dtype=self.dtype, device=self.device
            )
        else:
            positions = positions.to(dtype=self.dtype, device=self.device)

        natoms = self.batch.num_nodes
        new_atom_positions = positions[:natoms]
        
        # Handle exp_cell_factor division properly
        cell_positions = positions[natoms:]  # Shape: (3*batch_size, 3)
        new_log_deform_grad_list = []
        
        for i in range(len(self.get_cells())):
            # Get cell DOFs for this system
            cell_dofs = cell_positions[i*3:(i+1)*3]  # Shape: (3, 3)
            # Get scaling factor for this system
            scaling_factor = self.exp_cell_factor[i*3].item()
            
            # Follow ASE's FrechetCellFilter.set_positions logic exactly:
            # ASE logic: exmp(input / exp_cell_factor) gets passed to UnitCellFilter.set_positions
            # UnitCellFilter expects cell_factor * deformation_gradient as input
            # So: expm(input / exp_cell_factor) = cell_factor * deformation_gradient
            # We want to extract deformation_gradient for our implementation
            
            # 1. Divide by exp_cell_factor  
            log_deform_grad = cell_dofs / scaling_factor
            # 2. Apply expm to get what UnitCellFilter.set_positions expects
            log_deform_grad_np = log_deform_grad.cpu().detach().numpy()
            ucf_expected_input = self.expm(log_deform_grad_np)
            # 3. Divide by cell_factor (natoms) to get actual deformation gradient
            natoms_system = self.batch.natoms[i].item()
            deform_grad = ucf_expected_input / natoms_system
            deform_grad_tensor = torch.from_numpy(deform_grad).to(self.device)
            
            new_log_deform_grad_list.append(deform_grad_tensor)
        
        # new_log_deform_grad_list actually contains deformation gradients, not log gradients
        new_deform_grad = torch.stack(new_log_deform_grad_list)  # Shape: (batch_size, 3, 3)

        # Set the new cell from the original cell and the new deformation gradient
        new_cells = torch.bmm(
            self.orig_cells, torch.transpose(new_deform_grad, 1, 2)
        )
        self.set_cells(new_cells)

        # Set the positions from the ones passed in and the new deformation gradient
        new_atom_positions = torch.bmm(
            new_atom_positions.view(-1, 1, 3),
            torch.transpose(
                new_deform_grad[self.batch.batch, :, :].view(-1, 3, 3), 1, 2
            ),
        )
        super().set_positions(new_atom_positions.view(-1, 3))

    def get_potential_energy(self, **kwargs):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = super().get_potential_energy(**kwargs)
        return atoms_energy + self.pressure[0, 0] * self.get_volumes().sum()

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get forces and unit cell stress using Fréchet derivatives."""
        stress = self.get_property("stress", no_numpy=True).view(-1, 3, 3)
        atom_forces = self.get_property("forces", no_numpy=True)

        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            atom_forces[fixed_idx] = 0.0

        volumes = self.get_volumes().view(-1, 1, 1)
        virial = -volumes * (stress + self.pressure.view(-1, 3, 3))
        
        cur_deform_grad = self.deform_grad()
        atom_forces = torch.bmm(
            atom_forces.view(-1, 1, 3),
            cur_deform_grad[self.batch.batch, :, :].view(-1, 3, 3),
        )

        # Transform virial using current deformation gradient
        virial = torch.linalg.solve(
            cur_deform_grad, torch.transpose(virial, dim0=1, dim1=2)
        )
        virial = torch.transpose(virial, dim0=1, dim1=2)

        if self.hydrostatic_strain:
            virial = self._batch_diag(self._batch_trace(virial) / 3.0)

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask.view(-1, 3, 3)

        # Calculate Fréchet derivative-based cell forces
        deform_grad_log_force = torch.zeros_like(virial)
        cur_deform_grad_np = cur_deform_grad.cpu().detach().numpy()
        virial_np = virial.cpu().detach().numpy()
        
        for batch_idx in range(len(self.get_cells())):
            cur_deform_grad_log = self.logm(cur_deform_grad_np[batch_idx])
            ucf_cell_grad = virial_np[batch_idx] @ np.linalg.inv(cur_deform_grad_np[batch_idx].T)
            
            # Calculate Fréchet derivative for each component
            deform_grad_log_force_batch = np.zeros((3, 3))
            for mu in range(3):
                for nu in range(3):
                    dir_matrix = np.zeros((3, 3))
                    dir_matrix[mu, nu] = 1.0
                    # Directional derivative of deformation to (mu, nu) strain direction
                    expm_der = self.expm_frechet(
                        cur_deform_grad_log,
                        dir_matrix,
                        compute_expm=False
                    )
                    deform_grad_log_force_batch[mu, nu] = np.sum(expm_der * ucf_cell_grad)
            
            deform_grad_log_force[batch_idx] = torch.from_numpy(deform_grad_log_force_batch).to(self.device)

        if self.constant_volume:
            dglf_trace = self._batch_trace(deform_grad_log_force).view(-1, 1, 1)
            deform_grad_log_force -= self._batch_diag(dglf_trace.squeeze() / 3.0)

        natoms = self.batch.num_nodes
        augmented_forces = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            device=self.device,
            dtype=atom_forces.dtype,
        )
        
        augmented_forces[:natoms] = atom_forces.view(-1, 3)
        
        # Handle exp_cell_factor division properly for forces
        cell_forces = deform_grad_log_force.view(-1, 3)  # Shape: (3*batch_size, 3)
        cell_forces_scaled = torch.zeros_like(cell_forces)
        
        for i in range(len(self.get_cells())):
            # Get scaling factor for this system
            scaling_factor = self.exp_cell_factor[i*3].item()
            # Scale cell forces for this system
            cell_forces_scaled[i*3:(i+1)*3] = cell_forces[i*3:(i+1)*3] / scaling_factor
        
        augmented_forces[natoms:] = cell_forces_scaled

        self.stress = -virial.view(-1, 9) / volumes.view(-1, 1)

        if self.numpy and not no_numpy:
            augmented_forces = augmented_forces.cpu().numpy()

        return augmented_forces

    def __len__(self):
        return len(self.batch.pos) + 3 * len(self.batch)

    def get_potential_energies(self) -> torch.Tensor:
        """Get the predicted energy for each system in batch."""
        return (
            self.get_property("energy").view(-1)
            + self.pressure[0, 0] * self.get_volumes()
        )
