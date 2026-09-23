"""
batchASE: Batched ASE Relaxation Engine for high-throughput crystal structure prediction.
"""

import warnings

warnings.filterwarnings("ignore", message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")
warnings.filterwarnings("ignore", category=UserWarning, module="mace")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Please use atoms.calc = calc.*")

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
from .baseline import run_baseline
from .utils import count_atoms_cif, data_list_collater, ensure_directory

__version__ = "0.6.0"

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
    "run_baseline",
    "count_atoms_cif",
    "data_list_collater",
    "ensure_directory",
]
