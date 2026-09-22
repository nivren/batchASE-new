from .base import BatchPotential
from .mace import MACEBatchBackend
from .sevennet import SevenNetBatchBackend
from .chgnet import CHGNetBatchBackend
from .matris import MatRISBatchBackend


def create_backend(backend: str = "mace", **kwargs) -> BatchPotential:
    """Factory creating potential backend by model identifier."""
    b_name = backend.lower()
    if b_name == "mace":
        return MACEBatchBackend(**kwargs)
    elif b_name == "sevennet":
        return SevenNetBatchBackend(**kwargs)
    elif b_name == "chgnet":
        return CHGNetBatchBackend(**kwargs)
    elif b_name in ("matris", "matgl"):
        return MatRISBatchBackend(**kwargs)
    else:
        raise ValueError(f"Unknown potential backend: {backend}. Supported: ['mace', 'sevennet', 'chgnet', 'matris']")


__all__ = [
    "BatchPotential",
    "MACEBatchBackend",
    "SevenNetBatchBackend",
    "CHGNetBatchBackend",
    "MatRISBatchBackend",
    "create_backend",
]
