from pathlib import Path
from typing import Optional, Tuple

from .base import BatchPotential
from .mace import MACEBatchBackend
from .mock import MockBatchBackend
from .sevennet import SevenNetBatchBackend
from .chgnet import CHGNetBatchBackend
from .matris import MatRISBatchBackend


SUPPORTED_BACKENDS = ("mace", "mock")
UNIMPLEMENTED_BACKENDS = ("sevennet", "chgnet", "matris", "matgl")

#: Cache directory convention of the MACE foundation-model loaders.
MACE_CACHE_DIR = Path.home() / ".cache" / "mace"

#: MACE family aliases -> loader hint ("off" = MACE-OFF23, "mp" = MACE-MP).
_MACE_FAMILY_LOADERS = {
    "mace": "off",  # legacy alias, identical to "mace_off:small"
    "mace_off": "off",
    "mace_mp": "mp",
}


def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """Split a model identifier ``family[:spec]`` into family and spec.

    The family is case-insensitive; an empty spec after ':' yields None.
    """
    text = (model_id or "").strip()
    if not text:
        raise ValueError("Model identifier must not be empty")
    if ":" in text:
        family, spec = text.split(":", 1)
        family = family.strip().lower()
        spec = spec.strip() or None
        if not family:
            raise ValueError(f"Invalid model identifier '{model_id}': empty family before ':'")
        return family, spec
    return text.lower(), None


def resolve_checkpoint(spec: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a checkpoint spec to an existing local file, or return None.

    Search order:
      1. ``spec`` itself as a filesystem path (``~`` expanded).
      2. ``<cache_dir>/<spec>`` (default ``~/.cache/mace``), matching the
         caching convention of the MACE foundation-model loaders.

    Returning ``None`` means the spec is treated as a family-internal model
    name (e.g. ``"medium-mpa-0"``, ``"7net-0"``) and is left to the family's
    own loader, which may resolve or download it.
    """
    if not spec:
        return None
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate
    base = Path(cache_dir) if cache_dir is not None else MACE_CACHE_DIR
    cached = base / spec
    if cached.is_file():
        return cached
    return None


def _is_known_mace_name(loader: str, spec: str) -> bool:
    """Whether spec is a built-in model name of the given MACE loader family."""
    try:
        from mace.calculators.foundations_models import mace_mp_urls, mace_off_urls
    except ImportError:
        return False
    urls = mace_off_urls if loader == "off" else mace_mp_urls
    return spec in urls


def validate_model_id(model_id: str) -> Optional[Path]:
    """Validate a model identifier without instantiating any backend.

    Returns the resolved local checkpoint path (None when the family resolves
    its checkpoint internally, e.g. built-in names or no spec).

    Raises:
        ValueError: unknown family or malformed identifier.
        NotImplementedError: placeholder family (sevennet/chgnet/matris/matgl).
        FileNotFoundError: spec looks like a checkpoint file but was not found
            as a local path, in the MACE cache, nor among built-in names.
    """
    family, spec = parse_model_id(model_id)
    if family in UNIMPLEMENTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{family}' is currently a placeholder/under active development "
            f"and not yet functional in batchASE. Currently supported backends: {list(SUPPORTED_BACKENDS)}"
        )
    if family not in _MACE_FAMILY_LOADERS and family != "mock":
        raise ValueError(
            f"Unknown potential backend: '{family}'. Supported: {list(SUPPORTED_BACKENDS)}"
        )
    if family == "mock" or spec is None:
        return None
    resolved = resolve_checkpoint(spec)
    if resolved is not None:
        return resolved
    if _is_known_mace_name(_MACE_FAMILY_LOADERS[family], spec):
        return None
    raise FileNotFoundError(
        f"Checkpoint spec '{spec}' for family '{family}' was not found as a local path "
        f"('{Path(spec).expanduser()}') nor in the MACE cache ('{MACE_CACHE_DIR / spec}'), "
        f"and it is not a built-in {family} model name."
    )


def create_backend(model_id: str = "mace", **kwargs) -> BatchPotential:
    """Factory creating potential backend by model identifier.

    Identifier grammar: ``family[:spec]``. Supported families:

    - ``mace``            MACE-OFF23 small (legacy default, = ``mace_off:small``)
    - ``mace_off[:spec]`` MACE-OFF23 family (organic elements only); ``spec`` is
                          ``small``/``medium``/``large``, a cached model file name
                          (resolved in ``~/.cache/mace``), or a local path
    - ``mace_mp[:spec]``  MACE-MP family (89 elements incl. Li); ``spec`` is an
                          official name (e.g. ``medium-mpa-0``), a cached model
                          file name, or a local path (default: mace-mpa-0-medium)
    - ``mock``            Mock backend for tests

    Raises:
        NotImplementedError: If the requested family is a known placeholder
            under active development.
        ValueError: If the requested family is unknown.
    """
    family, spec = parse_model_id(model_id)

    if family in _MACE_FAMILY_LOADERS:
        loader_hint = kwargs.pop("loader", _MACE_FAMILY_LOADERS[family])
        model_value = kwargs.pop("model", None)
        if model_value is None and spec is not None:
            checkpoint = resolve_checkpoint(spec)
            # Bare built-in names (e.g. "medium-mpa-0") are forwarded to the
            # MACE loader which resolves/downloads them; resolved file paths
            # are used as-is.
            model_value = str(checkpoint) if checkpoint is not None else spec
        return MACEBatchBackend(model=model_value, loader=loader_hint, **kwargs)

    if family == "mock":
        return MockBatchBackend(**kwargs)

    if family in UNIMPLEMENTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{family}' is currently a placeholder/under active development "
            f"and not yet functional in batchASE. Currently supported backends: {list(SUPPORTED_BACKENDS)}"
        )
    raise ValueError(
        f"Unknown potential backend: '{family}'. Supported: {list(SUPPORTED_BACKENDS)}"
    )


__all__ = [
    "BatchPotential",
    "MACEBatchBackend",
    "MockBatchBackend",
    "SevenNetBatchBackend",
    "CHGNetBatchBackend",
    "MatRISBatchBackend",
    "SUPPORTED_BACKENDS",
    "UNIMPLEMENTED_BACKENDS",
    "MACE_CACHE_DIR",
    "parse_model_id",
    "resolve_checkpoint",
    "validate_model_id",
    "create_backend",
]
