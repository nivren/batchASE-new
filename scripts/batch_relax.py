#!/usr/bin/env python
"""
Unified CLI entry point for batch crystal structure relaxation (replaces mace_opt_batch.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import warnings

# Suppress known harmless upstream library warnings
warnings.filterwarnings("ignore", message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")

from batchase import Scheduler, ensure_directory, run_baseline
from batchase.engine.batching import order_structure_files
from batchase.potentials import validate_model_id


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch relaxation on molecular crystal structures.")
    parser.add_argument("--target_folder", type=str, required=True, help="Target folder containing CIF files")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--gpu_offset", type=int, default=0, help="Offset for GPU numbering")
    parser.add_argument("--batch_mode", choices=["bsize", "atoms"], default="bsize", help="bsize: balanced structure counts; atoms: first-fit packing by atom budget")
    parser.add_argument("--batch_size", type=int, default=4, help="Structure-count limit determining ceil(N/batch_size) balanced batches in bsize mode; ignored in atoms mode")
    parser.add_argument("--max_batch_atoms", type=int, default=0, help="Strict positive atom budget in atoms mode; ignored in bsize mode; optional validation limit for external fixed plans")
    parser.add_argument("--structure_order", choices=["rand", "syst", "atom"], default="rand", help="syst: first N files; rand: random N files in sample order; atom: the same random sample sorted by actual atom count descending")
    parser.add_argument("--structure_order_seed", type=int, default=42, help="Seed for rand sampling/order and atom sampling; syst ignores it")
    parser.add_argument("--fixed_batch_plan", type=str, default=None, help="Exact one-batch-per-worker JSON plan; no refill queue")
    parser.add_argument("--batch_plan_cache_dir", type=str, default=".cache/batch_plans", help="Directory for reusable internal batch plans; empty string or none disables caching; fixed plans bypass it")
    parser.add_argument("--batch_plan_cache_limit", type=int, default=20, help="Maximum number of cached plans in this directory; least recently used plans are evicted; must be positive")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum relaxation steps")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold (eV/A)")
    parser.add_argument("--fmax1", type=float, default=None, help="Force convergence threshold for Stage 1 (eV/A, defaults to --fmax)")
    parser.add_argument("--fmax2", type=float, default=None, help="Force convergence threshold for Stage 2 (eV/A, defaults to --fmax)")
    parser.add_argument("--filter1", type=str, default="UnitCellFilter", help="Filter for Stage 1 (UnitCellFilter or none)")
    parser.add_argument("--filter2", type=str, default=None, help="Filter for Stage 2 (UnitCellFilter or none)")
    parser.add_argument("--optimizer1", type=str, default="BFGSFusedLS", help="Optimizer for Stage 1")
    parser.add_argument("--optimizer2", type=str, default="BFGSFusedLS", help="Optimizer for Stage 2")
    parser.add_argument("--skip_second_stage", type=str2bool, nargs="?", const=True, default=False, help="Skip the second relaxation stage")
    parser.add_argument("--scalar_pressure", type=float, default=0.0006, help="External scalar pressure in eV/A^3 (0.0006 eV/A^3 ≈ 0.096 GPa / ~1000 bar)")
    parser.add_argument("--molecule_single", type=int, default=None, help="Reference atoms per single molecule for energy normalization")
    parser.add_argument("--output_path", type=str, default="./", help="Directory for output files")
    parser.add_argument(
        "--model",
        type=str,
        default="mace",
        help=(
            "Potential model identifier 'family[:spec]': mace (MACE-OFF23 small, legacy), "
            "mace_off[:small|medium|large|<file>], mace_mp[:<name|file>] (MACE-MP family, "
            "89 elements incl. Li; default mace-mpa-0-medium), mock. 'spec' may be an "
            "official model name, a file name under ~/.cache/mace, or a local path."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Execution device (for example 'cpu' or 'cuda:0'); overrides --n_gpus")
    parser.add_argument("--use_fasteq", type=str2bool, nargs="?", const=True, default=False, help="Enable FastEq acceleration")
    parser.add_argument("--cueq", type=str2bool, nargs="?", const=True, default=False, help="Enable cuEquivariance acceleration")
    parser.add_argument("--bfgs_cpu_thread", type=int, default=1, help="Threads for BFGS CPU eigh offload")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of CPU threads per process")
    parser.add_argument("--bind_cores", type=str, default=None, help="Core ranges for each worker")
    parser.add_argument("--compile_mode", type=str, default=None, help="torch.compile mode")
    parser.add_argument("--profile", type=str, default="False", help="Enable profiling options")
    parser.add_argument("--num_structures", type=int, default=0, help="Maximum structures to optimize (0 = all); structure_order controls selection and order")
    parser.add_argument("--run_baseline", type=str2bool, nargs="?", const=True, default=False, help="Run baseline sequential ASE")
    parser.add_argument("--log_level", type=lambda s: s.upper(), default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_structures < 0:
        logging.error("--num_structures must be nonnegative")
        return 1

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Preflight: fail fast on unknown model families or missing local
    # checkpoints, before spawning any worker process.
    try:
        resolved_ckpt = validate_model_id(args.model)
    except (ValueError, NotImplementedError, FileNotFoundError) as exc:
        logging.error(f"Invalid --model '{args.model}': {exc}")
        return 1
    if resolved_ckpt is not None:
        logging.info(f"Model '{args.model}' resolved to checkpoint: {resolved_ckpt}")

    target_folder = pathlib.Path(args.target_folder)
    files = sorted([str(f) for f in target_folder.glob("*.cif")])
    if not files:
        logging.error(f"No CIF files found in {target_folder}")
        return 1

    total_found = len(files)
    selected_count = min(args.num_structures, total_found) if args.num_structures else total_found
    logging.info("batchASE: Using %s / %s structures (order=%s, seed=%s)",
                 selected_count, total_found, args.structure_order, args.structure_order_seed)

    output_path = os.path.abspath(args.output_path)
    ensure_directory(output_path)

    devices = [args.device] if args.device else [
        f"cuda:{i}" for i in range(args.gpu_offset, args.gpu_offset + args.n_gpus)
    ]

    fmax1 = args.fmax1 if args.fmax1 is not None else args.fmax
    fmax2 = args.fmax2 if args.fmax2 is not None else args.fmax

    logging.info(f"Target devices: {devices}, Workers: {args.num_workers}, Batch size: {args.batch_size}")
    logging.info(f"Optimizers: Stage1={args.optimizer1} (filter={args.filter1}, fmax={fmax1}), Stage2={args.optimizer2} (filter={args.filter2}, fmax={fmax2})")

    filter1 = None if args.filter1 in ("none", None, "None") else args.filter1
    filter2 = None if args.filter2 in ("none", None, "None") else args.filter2

    if args.run_baseline:
        files = order_structure_files(files, args.structure_order, args.structure_order_seed,
                                      num_structures=args.num_structures)
        with open(os.path.join(output_path, "manifest.txt"), "w") as f:
            f.write("\n".join(files) + "\n")
        logging.info("Running baseline sequential ASE optimization...")
        run_baseline(
            files=files,
            num_workers=args.num_workers,
            devices=devices,
            max_steps=args.max_steps,
            filter1=filter1,
            filter2=filter2,
            skip_second_stage=args.skip_second_stage,
            scalar_pressure=args.scalar_pressure,
            optimizer1=args.optimizer1,
            optimizer2=args.optimizer2,
            fmax=args.fmax,
            fmax1=fmax1,
            fmax2=fmax2,
            output_path=output_path,
            model=args.model,
        )
        logging.info("Baseline relaxation completed.")
        return

    scheduler = Scheduler(
        files=files,
        num_workers=args.num_workers,
        devices=devices,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        fmax=args.fmax,
        fmax1=fmax1,
        fmax2=fmax2,
        filter1=filter1,
        filter2=filter2,
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
        max_batch_atoms=args.max_batch_atoms,
        batch_mode=args.batch_mode,
        structure_order=args.structure_order,
        structure_order_seed=args.structure_order_seed,
        fixed_batch_plan=args.fixed_batch_plan,
        batch_plan_cache_dir=args.batch_plan_cache_dir,
        batch_plan_cache_limit=args.batch_plan_cache_limit,
        num_structures=args.num_structures,
        num_threads=args.num_threads,
        bind_cores=args.bind_cores,
        compile_mode=args.compile_mode,
        profile=args.profile,
    )
    try:
        scheduler.run()
    except (ValueError, OSError, RuntimeError) as exc:
        logging.error(f"Batch relaxation failed: {exc}")
        return 1
    logging.info("Batch relaxation workflow completed.")


if __name__ == "__main__":
    raise SystemExit(main())
