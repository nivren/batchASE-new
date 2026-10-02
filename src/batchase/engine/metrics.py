"""Aggregate completed batch metrics without averaging percentages."""

from __future__ import annotations


def merge_stage_metrics(aggregate, metrics):
    """Sum counts/times and retain peaks, worst structures, and profiler totals."""
    count_keys = ("structures", "converged", "output_cifs", "steps", "batch_iterations")
    time_keys = ("elapsed_s", "mace_s", "opt_s", "graph_s", "io_s")
    for key in count_keys + time_keys:
        aggregate[key] = aggregate.get(key, 0) + metrics.get(key, 0)
    for key in ("max_observed_batch_atoms", "peak_vram_gb"):
        aggregate[key] = max(aggregate.get(key, 0), metrics.get(key, 0))
    aggregate["max_batch_atoms"] = metrics.get("max_batch_atoms", 0)
    if metrics.get("max_structure_steps", 0) >= aggregate.get("max_structure_steps", 0):
        for key in ("max_structure_steps", "max_structure_file", "max_structure_status",
                    "max_structure_failed_reason", "max_structure_fmax"):
            aggregate[key] = metrics.get(key)

    breakdown = metrics.get("profiler_breakdown")
    if breakdown is None:
        return
    combined = aggregate.setdefault("profiler_breakdown", {"enabled": False, "calls": {}})
    combined["enabled"] = combined["enabled"] or breakdown.get("enabled", False)
    for key, seconds in breakdown.items():
        if key.endswith("_s"):
            combined[key] = combined.get(key, 0.0) + seconds
    for key, count in breakdown.get("calls", {}).items():
        combined["calls"][key] = combined["calls"].get(key, 0) + count
    measured = combined.get("measured_total_s", 0.0)
    wall = combined.get("optimizer_wall_s", 0.0)
    combined["unprofiled_optimizer_s"] = max(wall - measured, 0.0)
    for key, seconds in list(combined.items()):
        if key.endswith("_s"):
            name = key[:-2]
            combined[f"{name}_ms"] = seconds * 1000.0
            if name not in ("measured_total", "optimizer_wall", "unprofiled_optimizer"):
                combined[f"{name}_pct"] = 100.0 * seconds / measured if measured else 0.0
    combined["profiler_coverage_pct"] = 100.0 * measured / wall if wall else 0.0
    combined["unprofiled_optimizer_pct"] = (
        100.0 * combined["unprofiled_optimizer_s"] / wall if wall else 0.0
    )
