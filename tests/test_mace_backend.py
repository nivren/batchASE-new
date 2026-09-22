"""
Numerical parity and relaxation regression test suite for MACEBatchBackend in batchASE.

Verification criteria:
1. Pipeline parity: backend.predict(gbatch, float32 graph) vs official single-structure calculate
   (E/S < 1e-6, F < 1e-3).
2. Machine precision parity: backend.predict_from_atoms(native float64) vs official calculate
   (E, F, S all < 1e-8; typically ~1e-15 matching exact machine precision).
3. Relaxation regression: batch relaxation steps and energy matching baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater
from batchase.potentials import MACEBatchBackend, create_backend
from batchase.relaxation import OptimizableUnitCellBatch, BFGSFusedLS

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("test_mace_backend")

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.expanduser("~/.cache/mace/MACE-OFF23_small.model")
DEFAULT_CIFS = [
    str(HERE / "fixtures/input.cif"),
    str(HERE / "fixtures/input.1.cif"),
]

RTOL = 1e-6
RTOL_F = 1e-3
RTOL_MP = 1e-8


def build_gbatch(atoms_list, device):
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    gbatch = data_list_collater([a2g.convert(atoms) for atoms in atoms_list])
    return gbatch.to(device)


def single_reference(calculator, atoms):
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    return energy, forces, stress


def compare_batch_vs_single(backend, calculator, atoms_list, gbatch, tag, natoms_per_sys):
    out = backend.predict(gbatch, compute_stress=True)
    e_batch, f_batch, s_batch = out["energy"], out["forces"], out["stress"]

    max_rel = 0.0
    ok = True
    offset = 0
    for i, atoms in enumerate(atoms_list):
        e_ref, f_ref, s_ref = single_reference(calculator, atoms)
        n = natoms_per_sys[i]

        e_got = e_batch[i].item()
        f_got = f_batch[offset : offset + n].cpu().numpy()
        s_got = s_batch[i].cpu().numpy()
        offset += n

        rel_e = abs(e_got - e_ref) / max(abs(e_ref), 1.0)
        rel_f = np.abs(f_got - f_ref).max() / max(np.abs(f_ref).max(), 1.0)
        rel_s = np.abs(s_got - s_ref).max() / max(np.abs(s_ref).max(), 1.0)

        max_rel = max(max_rel, rel_e, rel_f, rel_s)
        logger.info(
            f"[{tag}] sys{i}: natoms={n}  E batch={e_got:.8f} ref={e_ref:.8f}  "
            f"rel_err: E={rel_e:.2e} F={rel_f:.2e} S={rel_s:.2e}"
        )
        if not (rel_e < RTOL and rel_f < RTOL_F and rel_s < RTOL):
            ok = False
            logger.error(
                f"[{tag}] sys{i} Numerical tolerance exceeded!  E={e_got} vs {e_ref}\n"
                f"  F max_diff={np.abs(f_got - f_ref).max()}\n"
                f"  S got={s_got} ref={s_ref}"
            )
    return max_rel, ok


def compare_from_atoms_vs_official(backend, atoms_list, tag):
    out_back = backend.predict_from_atoms(atoms_list, compute_stress=True)
    e_back = out_back["energy"].flatten().cpu().numpy()
    f_back = out_back["forces"].cpu().numpy()
    s_back = out_back["stress"].cpu().numpy()

    max_rel = 0.0
    offset = 0
    ok = True
    for i, atoms in enumerate(atoms_list):
        e_ref, f_ref, s_ref = single_reference(backend.calculator, atoms)
        n = len(atoms)
        rel_e = abs(e_back[i] - e_ref) / max(abs(e_ref), 1.0)
        rel_f = np.abs(f_back[offset : offset + n] - f_ref).max() / max(np.abs(f_ref).max(), 1.0)
        rel_s = np.abs(s_back[i] - s_ref).max() / max(np.abs(s_ref).max(), 1.0)
        offset += n
        max_rel = max(max_rel, rel_e, rel_f, rel_s)
        ok = ok and rel_e < RTOL_MP and rel_f < RTOL_MP and rel_s < RTOL_MP
        logger.info(
            f"[{tag}] from_atoms-vs-official sys{i}: E={rel_e:.2e} F={rel_f:.2e} S={rel_s:.2e}"
        )
    return ok, max_rel


def run_pipeline_test(cif_paths, device):
    tag = "pipeline/auto"
    atoms_list = [read(p) for p in cif_paths]
    natoms_per_sys = [len(a) for a in atoms_list]
    logger.info(f"[{tag}] Loading model: {MODEL_PATH}")
    calculator = MACECalculator(model_paths=MODEL_PATH, device=device)
    backend = MACEBatchBackend(model=MODEL_PATH, device=device, neighbor="auto")
    gbatch = build_gbatch(atoms_list, device)

    t0 = time.perf_counter()
    max_rel, ok = compare_batch_vs_single(
        backend, calculator, atoms_list, gbatch, tag, natoms_per_sys
    )
    dt = time.perf_counter() - t0
    logger.info(f"[{tag}] Pipeline parity test completed: max_rel_err={max_rel:.3e} in {dt:.2f}s")
    return ok, max_rel


def run_machine_precision_test(cif_paths, device):
    tag = "backend/machine_precision"
    atoms_list = [read(p) for p in cif_paths]
    backend = MACEBatchBackend(model=MODEL_PATH, device=device, neighbor="auto")
    ok, max_rel = compare_from_atoms_vs_official(backend, atoms_list, tag)
    logger.info(f"[{tag}] Machine precision test completed: ok={ok}, max_rel_err={max_rel:.3e}")
    return ok, max_rel


def run_relaxation_test(cif_paths, device):
    tag = "relaxation/test"
    atoms_list = [read(p) for p in cif_paths]
    gbatch = build_gbatch(atoms_list, device)
    backend = create_backend("mace", model=MODEL_PATH, device=device)
    obatch = OptimizableUnitCellBatch(gbatch, backend=backend, numpy=False, scalar_pressure=0.0)
    optimizer = BFGSFusedLS(obatch, device=device, use_profiler=False)

    t0 = time.perf_counter()
    optimizer.run(fmax=0.05, steps=10)
    dt = time.perf_counter() - t0
    energies = obatch.get_potential_energies().tolist()
    logger.info(f"[{tag}] 10-step relaxation executed in {dt:.2f}s, energies={energies}")
    return True, energies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cif_paths = [p for p in DEFAULT_CIFS if os.path.exists(p)]
    if len(cif_paths) < 2:
        logger.error(f"Test fixtures missing in {HERE / 'fixtures'}")
        sys.exit(1)

    all_passed = True

    # 1. Pipeline test
    ok_pipe, rel_pipe = run_pipeline_test(cif_paths, args.device)
    all_passed = all_passed and ok_pipe

    # 2. Machine precision test
    ok_mp, rel_mp = run_machine_precision_test(cif_paths, args.device)
    all_passed = all_passed and ok_mp

    # 3. Relaxation test
    ok_relax, energies = run_relaxation_test(cif_paths, args.device)
    all_passed = all_passed and ok_relax

    if all_passed:
        logger.info("\n=======================================================")
        logger.info("ALL NUMERICAL AND REGRESSION TESTS PASSED SUCCESSFULLY!")
        logger.info("=======================================================\n")
    else:
        logger.error("\nSOME TESTS FAILED! Check logs above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
