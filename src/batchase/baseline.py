"""
Baseline sequential ASE relaxation implementation for benchmark comparison.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from ase.io import read
from ase.optimize import LBFGS as ASE_LBFGS
from ase.optimize import QuasiNewton as ASE_QuasiNewton
from ase.optimize import BFGS as ASE_BFGS
from ase.optimize import FIRE as ASE_FIRE
from ase.optimize.fire2 import FIRE2 as ASE_FIRE2

from .utils import ensure_directory

logger = logging.getLogger("batchase.baseline")


def baseline_task(
    file: str,
    device: str,
    max_steps: int,
    filter1: Optional[str] = "UnitCellFilter",
    filter2: Optional[str] = None,
    skip_second_stage: bool = False,
    scalar_pressure: float = 0.0006,
    first_optimizer: str = "LBFGS",
    second_optimizer: str = "LBFGS",
    fmax: float = 0.01,
    fmax1: Optional[float] = None,
    fmax2: Optional[float] = None,
    output_path: str = "./",
    model: str = "mace",
) -> dict:
    """
    Run baseline sequential ASE optimization on a single crystal structure file.
    """
    if str(device).startswith("cuda"):
        import torch
        dev_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        try:
            torch.cuda.set_device(dev_idx)
        except Exception:
            pass

    press_dir = os.path.join(output_path, "cif_result_press")
    final_dir = os.path.join(output_path, "cif_result_final")
    ensure_directory(press_dir)
    ensure_directory(final_dir)

    stem = Path(file).stem
    crystal = read(file)

    if model == "mace":
        from mace.calculators import mace_off
        calc = mace_off(model="small", device="cuda" if str(device).startswith("cuda") else "cpu")
    else:
        from .potentials import create_backend
        backend = create_backend(model, device=device)
        calc = backend.calculator

    crystal.calc = calc

    target_fmax1 = fmax1 if fmax1 is not None else fmax
    target_fmax2 = fmax2 if fmax2 is not None else fmax

    optimizer_map = {
        "lbfgs": ASE_LBFGS,
        "quasinewton": ASE_QuasiNewton,
        "bfgs": ASE_BFGS,
        "bfgsfusedls": ASE_BFGS,
        "bfgslinesearch": ASE_BFGS,
        "fire": ASE_FIRE,
        "fire2": ASE_FIRE2,
    }
    first_opt_cls = optimizer_map.get(str(first_optimizer).lower(), ASE_LBFGS)
    second_opt_cls = optimizer_map.get(str(second_optimizer).lower(), ASE_LBFGS)

    # Stage 1
    if filter1 == "UnitCellFilter":
        from ase.filters import UnitCellFilter
        atoms_stage1 = UnitCellFilter(crystal, scalar_pressure=scalar_pressure)
        opt1 = first_opt_cls(atoms_stage1)
    elif filter1 == "FrechetCellFilter":
        from ase.filters import FrechetCellFilter
        atoms_stage1 = FrechetCellFilter(crystal, scalar_pressure=scalar_pressure)
        opt1 = first_opt_cls(atoms_stage1)
    else:
        opt1 = first_opt_cls(crystal)

    t0_s1 = time.perf_counter()
    opt1.run(fmax=target_fmax1, steps=max_steps)
    t1_s1 = time.perf_counter()
    s1_time = t1_s1 - t0_s1
    s1_steps = getattr(opt1, "nsteps", 0)

    output_press = os.path.join(press_dir, f"{stem}.cif")
    crystal.write(output_press)

    if skip_second_stage:
        return {
            "file": stem,
            "stage1_time": s1_time,
            "stage1_steps": s1_steps,
            "stage2_time": 0.0,
            "stage2_steps": 0,
            "total_time": s1_time,
            "total_steps": s1_steps,
        }

    # Stage 2
    crystal = read(output_press)
    crystal.calc = calc

    if filter2 == "UnitCellFilter":
        from ase.filters import UnitCellFilter
        atoms_stage2 = UnitCellFilter(crystal)
        opt2 = second_opt_cls(atoms_stage2)
    elif filter2 == "FrechetCellFilter":
        from ase.filters import FrechetCellFilter
        atoms_stage2 = FrechetCellFilter(crystal)
        opt2 = second_opt_cls(atoms_stage2)
    else:
        opt2 = second_opt_cls(crystal)

    t0_s2 = time.perf_counter()
    opt2.run(fmax=target_fmax2, steps=max_steps)
    t1_s2 = time.perf_counter()
    s2_time = t1_s2 - t0_s2
    s2_steps = getattr(opt2, "nsteps", 0)

    output_final = os.path.join(final_dir, f"{stem}.cif")
    crystal.write(output_final)

    return {
        "file": stem,
        "stage1_time": s1_time,
        "stage1_steps": s1_steps,
        "stage2_time": s2_time,
        "stage2_steps": s2_steps,
        "total_time": s1_time + s2_time,
        "total_steps": s1_steps + s2_steps,
    }


def run_baseline(
    files: List[str],
    num_workers: int = 1,
    devices: Optional[List[str]] = None,
    max_steps: int = 100,
    filter1: Optional[str] = "UnitCellFilter",
    filter2: Optional[str] = None,
    skip_second_stage: bool = False,
    scalar_pressure: float = 0.0006,
    optimizer1: str = "LBFGS",
    optimizer2: str = "LBFGS",
    fmax: float = 0.01,
    fmax1: Optional[float] = None,
    fmax2: Optional[float] = None,
    output_path: str = "./",
    model: str = "mace",
) -> None:
    """
    Run baseline standard ASE relaxation across multiple worker processes.
    """
    devices = devices or ["cuda:0"]
    logger.info(f"Starting baseline optimization with {num_workers} workers on devices: {devices}")
    output_path = os.path.abspath(output_path)
    ensure_directory(output_path)

    start_time = time.perf_counter()

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=num_workers)(
            delayed(baseline_task)(
                file=file,
                device=devices[i % len(devices)],
                max_steps=max_steps,
                filter1=filter1,
                filter2=filter2,
                skip_second_stage=skip_second_stage,
                scalar_pressure=scalar_pressure,
                first_optimizer=optimizer1,
                second_optimizer=optimizer2,
                fmax=fmax,
                fmax1=fmax1,
                fmax2=fmax2,
                output_path=output_path,
                model=model,
            )
            for i, file in enumerate(files)
        )
    except ImportError:
        import multiprocessing as mp
        with mp.Pool(processes=num_workers) as pool:
            tasks = [
                (
                    file,
                    devices[i % len(devices)],
                    max_steps,
                    filter1,
                    filter2,
                    skip_second_stage,
                    scalar_pressure,
                    optimizer1,
                    optimizer2,
                    fmax,
                    fmax1,
                    fmax2,
                    output_path,
                    model,
                )
                for i, file in enumerate(files)
            ]
            results = pool.starmap(baseline_task, tasks)

    total_elapsed = time.perf_counter() - start_time
    logger.info(f"Baseline optimization finished in {total_elapsed:.2f}s")

    csv_file = os.path.join(output_path, "results_baseline.csv")
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "stage1_time",
                "stage1_steps",
                "stage2_time",
                "stage2_steps",
                "total_time",
                "total_steps",
            ],
        )
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    summary_file = os.path.join(output_path, "summary_baseline.csv")
    with open(summary_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_time", "num_workers", "batch_size"])
        writer.writeheader()
        writer.writerow({
            "elapsed_time": total_elapsed,
            "num_workers": num_workers,
            "batch_size": 1,
        })
