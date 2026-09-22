"""
Optimizer registry and factory.
"""

from __future__ import annotations

from typing import Type
from .base import BatchOptimizer
from .bfgs import BFGS
from .bfgsfusedls import BFGSFusedLS
from .lbfgs import LBFGS

_OPTIMIZERS: dict[str, Type] = {
    "bfgs": BFGS,
    "bfgsfusedls": BFGSFusedLS,
    "lbfgs": LBFGS,
    "quasinewton": BFGS,
}


def register_optimizer(name: str, opt_cls: Type):
    """Register a new batch optimizer class."""
    _OPTIMIZERS[name.lower()] = opt_cls


def get_optimizer_cls(name: str) -> Type:
    """Retrieve optimizer class by case-insensitive name."""
    clean_name = name.lower()
    if clean_name not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Registered: {list(_OPTIMIZERS.keys())}"
        )
    return _OPTIMIZERS[clean_name]
