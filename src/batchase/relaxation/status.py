"""
Slot status and failure reason definitions for batched relaxation.
"""

from enum import Enum


class SlotStatus(str, Enum):
    ACTIVE = "active"
    CONVERGED = "converged"
    FAILED = "failed"


class FailReason:
    NAN_FORCE = "nan_force"
    INF_FORCE = "inf_force"
    FORCE_OVERFLOW = "force_overflow"
    INVALID_CELL = "invalid_cell"
    MAX_STEPS = "max_steps"
    EIGENSOLVER_FAILED = "eigensolver_failed"
