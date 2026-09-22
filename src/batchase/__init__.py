"""
batchASE: Batched ASE Relaxation Engine for high-throughput crystal structure prediction.
"""

from .neighbors import AtomsToGraphs
from .kernels import detect_optimal_backend, get_pbc_graph_kernel
from .potentials import BatchPotential, MACEBatchBackend, create_backend
from .relaxation import (
    OptimizableBatch,
    OptimizableUnitCellBatch,
    OptimizableFrechetCellBatch,
    BatchOptimizer,
    BFGS,
    BFGSFusedLS,
    LBFGS,
    get_optimizer_cls,
    register_optimizer,
)
from .engine import SlotManager, Worker, Scheduler
from .utils import count_atoms_cif, data_list_collater, ensure_directory

__version__ = "0.5.0"

__all__ = [
    "AtomsToGraphs",
    "detect_optimal_backend",
    "get_pbc_graph_kernel",
    "BatchPotential",
    "MACEBatchBackend",
    "create_backend",
    "OptimizableBatch",
    "OptimizableUnitCellBatch",
    "OptimizableFrechetCellBatch",
    "BatchOptimizer",
    "BFGS",
    "BFGSFusedLS",
    "LBFGS",
    "get_optimizer_cls",
    "register_optimizer",
    "SlotManager",
    "Worker",
    "Scheduler",
    "count_atoms_cif",
    "data_list_collater",
    "ensure_directory",
]
