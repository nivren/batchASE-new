from .base import BatchPotential
from .mace import MACEBatchBackend
from .mock import MockBatchBackend
from .sevennet import SevenNetBatchBackend
from .chgnet import CHGNetBatchBackend
from .matris import MatRISBatchBackend


SUPPORTED_BACKENDS = ("mace", "mock")
UNIMPLEMENTED_BACKENDS = ("sevennet", "chgnet", "matris", "matgl")


def create_backend(backend: str = "mace", **kwargs) -> BatchPotential:
    """Factory creating potential backend by model identifier.

    Raises:
        NotImplementedError: If the requested backend is a known placeholder under active development.
        ValueError: If the requested backend is unknown.
    """
    b_name = backend.lower()
    if b_name in SUPPORTED_BACKENDS:
        if b_name == "mace":
            return MACEBatchBackend(**kwargs)
        if b_name == "mock":
            return MockBatchBackend(**kwargs)
    elif b_name in UNIMPLEMENTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{backend}' is currently a placeholder/under active development "
            f"and not yet functional in batchASE. Currently supported backends: {list(SUPPORTED_BACKENDS)}"
        )
    else:
        raise ValueError(
            f"Unknown potential backend: '{backend}'. Supported: {list(SUPPORTED_BACKENDS)}"
        )


__all__ = [
    "BatchPotential",
    "MACEBatchBackend",
    "MockBatchBackend",
    "SevenNetBatchBackend",
    "CHGNetBatchBackend",
    "MatRISBatchBackend",
    "create_backend",
]
