#!/usr/bin/env python
"""
Unified CLI entry point for batch crystal structure relaxation (replaces mace_opt_batch.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time

from batchase import Scheduler, ensure_directory, count_atoms_cif


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch relaxation on molecular crystal structures.")
    parser.add_argument("--target_folder", type=str, required=True, help="Target folder containing CIF files")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--gpu_offset", type=int, default=0, help="Offset for GPU numbering")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per worker")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum relaxation steps")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold (eV/A)")
    parser.add_argument("--filter1", type=str, default="UnitCellFilter", choices=[None, "UnitCellFilter"], help="Filter for Stage 1")
    parser.add_argument("--filter2", type=str, default=None, choices=[None, "UnitCellFilter"], help="Filter for Stage 2")
    parser.add_argument("--optimizer1", type=str, default="BFGSFusedLS", help="Optimizer for Stage 1")
    parser.add_argument("--optimizer2", type=str, default="BFGSFusedLS", help="Optimizer for Stage 2")
    parser.add_argument("--skip_second_stage", action="store_true", help="Skip the second relaxation stage")
    parser.add_argument("--scalar_pressure", type=float, default=0.0006, help="External scalar pressure (GPa/unit)")
    parser.add_argument("--molecule_single", type=int, default=64, help="Reference atoms per molecule")
    parser.add_argument("--output_path", type=str, default="./", help="Directory for output files")
    parser.add_argument("--model", type=str, default="mace", choices=["mace", "sevennet", "chgnet", "matris"], help="MLIP model backend")
    parser.add_argument("--use_ordered_files", action="store_true", help="Sort CIF files by atomic count descending")
    parser.add_argument("--use_fasteq", action="store_true", help="Enable FastEq acceleration")
    parser.add_argument("--cueq", action="store_true", help="Enable cuEquivariance acceleration")
    parser.add_argument("--bfgs_cpu_thread", type=int, default=1, help="Threads for BFGS CPU eigh offload")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    target_folder = pathlib.Path(args.target_folder)
    files = [str(f) for f in target_folder.glob("*.cif")]
    if not files:
        logging.error(f"No CIF files found in {target_folder}")
        return

    output_path = os.path.abspath(args.output_path)
    ensure_directory(output_path)

    devices = [f"cuda:{i}" for i in range(args.gpu_offset, args.gpu_offset + args.n_gpus)]

    logging.info(f"batchASE: Found {len(files)} files in {target_folder}")
    logging.info(f"Target devices: {devices}, Workers: {args.num_workers}, Batch size: {args.batch_size}")
    logging.info(f"Optimizers: Stage1={args.optimizer1} (filter={args.filter1}), Stage2={args.optimizer2} (filter={args.filter2})")

    # Record manifest
    with open(os.path.join(output_path, "manifest.txt"), "w") as f:
        f.write("\n".join(files) + "\n")

    if args.use_ordered_files:
        logging.info("Sorting structures by atom count (descending)...")
        t0 = time.perf_counter()
        files = sorted(files, key=count_atoms_cif, reverse=True)
        logging.info(f"Sorting completed in {time.perf_counter() - t0:.3f}s")

    scheduler = Scheduler(
        files=files,
        num_workers=args.num_workers,
        devices=devices,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        fmax=args.fmax,
        filter1=args.filter1,
        filter2=args.filter2,
        optimizer1=args.optimizer1,
        optimizer2=args.optimizer2,
        skip_second_stage=args.skip_second_stage,
        scalar_pressure=args.scalar_pressure,
        molecule_single=args.molecule_single,
        output_path=output_path,
        model=args.model,
        use_fasteq=args.use_fasteq,
        cueq=args.cueq,
        bfgs_cpu_thread=args.bfgs_cpu_thread,
    )
    scheduler.run()
    logging.info("Batch relaxation workflow completed.")


if __name__ == "__main__":
    main()
