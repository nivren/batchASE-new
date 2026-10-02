"""Reuse complete batch plans while checking input metadata and effective options."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import ase

from . import batching

logger = logging.getLogger(__name__)
CACHE_VERSION = 2


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _cache_key(paths, parameters):
    inputs = []
    # Random sample order is part of the plan; atom ties use filename order.
    key_paths = paths if parameters["structure_order"] == "rand" else sorted(paths)
    for path in key_paths:
        stat = os.stat(path)
        inputs.append((path, stat.st_size, stat.st_mtime_ns))
    return _digest({
        "cache_version": CACHE_VERSION,
        "planning_version": batching.PLANNING_VERSION,
        "ase_version": ase.__version__,
        "python_version": list(sys.version_info[:2]),
        "parameters": parameters,
        "inputs": inputs,
    })


@contextmanager
def _cache_lock(directory):
    """Serialize publication and eviction across native Unix processes."""
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".cache.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prune_cache(directory, limit):
    entries = []
    for path in directory.glob("*.json"):
        # Only remove plans owned by this cache, never unrelated JSON files.
        if len(path.stem) != 64 or any(c not in "0123456789abcdef" for c in path.stem):
            continue
        try:
            entries.append((path.stat().st_mtime_ns, path.name, path))
        except FileNotFoundError:
            continue
    for _, _, path in sorted(entries)[:max(0, len(entries) - limit)]:
        path.unlink(missing_ok=True)


def _touch_cache(path, limit):
    with _cache_lock(path.parent):
        try:
            os.utime(path, None)
        except FileNotFoundError:
            pass  # Another process may have evicted this already-loaded plan.
        _prune_cache(path.parent, limit)


def _write_cache(path, key, plan):
    """Concurrent planners publish complete JSON files with atomic replacement."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".plan-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({"cache_version": CACHE_VERSION, "key": key,
                       "plan_sha256": _digest(plan), "plan": plan}, handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_or_build_batch_plan(files, batch_mode="bsize", batch_size=4, max_batch_atoms=0,
                             structure_order="rand", structure_order_seed=42, cache_dir=None,
                             cache_limit=20, num_structures=0):
    """Return (plan, cache report); cached paths are absolute and run-independent.

    Metadata validation avoids reading CIF contents. A file modified while
    preserving both its size and modification time is outside this contract.
    """
    started = time.perf_counter()
    batching.validate_batch_parameters(batch_mode, batch_size, max_batch_atoms, structure_order)
    candidates = [os.path.abspath(os.fspath(path)) for path in files]
    paths = batching.select_structure_files(candidates, structure_order,
                                             structure_order_seed, num_structures)
    options = dict(batch_mode=batch_mode, batch_size=batch_size,
                   max_batch_atoms=max_batch_atoms, structure_order=structure_order,
                   structure_order_seed=structure_order_seed)
    enabled = cache_dir is not None and os.fspath(cache_dir).lower() not in ("", "none")
    path = None
    status = "disabled"
    if enabled:
        if not isinstance(cache_limit, int) or cache_limit <= 0:
            raise ValueError("batch_plan_cache_limit must be a positive integer")
        parameters = {
            "batch_mode": batch_mode,
            "batch_size": batch_size if batch_mode == "bsize" else None,
            "max_batch_atoms": max_batch_atoms if batch_mode == "atoms" else 0,
            "structure_order": structure_order,
            "structure_order_seed": (structure_order_seed if structure_order == "rand"
                                      or (structure_order == "atom" and len(paths) < len(candidates))
                                      else None),
            "num_structures": len(paths),
        }
        key = _cache_key(paths, parameters)
        path = Path(cache_dir).expanduser().absolute() / f"{key}.json"
        status = "miss"
        try:
            with path.open(encoding="utf-8") as handle:
                cached = json.load(handle)
            plan = cached["plan"]
            if (cached["cache_version"] != CACHE_VERSION or cached["key"] != key
                    or cached["plan_sha256"] != _digest(plan)
                    or plan["planning_version"] != batching.PLANNING_VERSION):
                raise ValueError("Cache version or checksum mismatch")
            # Reflect current inactive settings without invalidating the grouping.
            plan["structure_order_seed"] = structure_order_seed
            plan.update(num_candidates=len(candidates), num_structures=num_structures)
            try:
                _touch_cache(path, cache_limit)
            except OSError as exc:
                logger.warning("Cannot maintain batch plan cache %s: %s; using cached plan", path, exc)
            return plan, {"status": "hit", "path": str(path),
                          "elapsed_s": time.perf_counter() - started}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Cannot reuse batch plan cache %s: %s; rebuilding", path, exc)

    plan = batching._build_selected_batch_plan(paths, **options)
    plan.update(num_candidates=len(candidates), num_structures=num_structures)
    if enabled:
        if _cache_key(paths, parameters) != key:
            status = "inputs_changed"
            logger.warning("Inputs changed during planning; this plan will not be cached")
        else:
            try:
                with _cache_lock(path.parent):
                    _write_cache(path, key, plan)
                    _prune_cache(path.parent, cache_limit)
            except OSError as exc:
                status = "unavailable"
                logger.warning("Cannot write batch plan cache %s: %s; using fresh plan", path, exc)
    return plan, {"status": status, "path": str(path) if path else None,
                  "elapsed_s": time.perf_counter() - started}
