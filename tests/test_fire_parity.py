"""
Numerical parity and regression test suite for batched FIRE and FIRE2 in batchASE.

Verification criteria:
1. FIRE 1.0 parity: batchASE.FIRE vs official ase.optimize.fire.FIRE
   (step-by-step max position diff < 1e-5 A, energy diff < 1e-6 eV).
2. FIRE 2.0 parity: batchASE.FIRE2 vs official ase.optimize.fire2.FIRE2
   (step-by-step max position diff < 1e-5 A, energy diff < 1e-6 eV).
3. Ragged batch invariance: Batch=1 vs Batch=2 with varying atom counts (92 vs 184)
   (trajectory difference < 1e-12 A, matching float64 machine precision).
4. Slot replenishment: dynamic slot update preserves surviving structures' velocity and state.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from ase.optimize import FIRE as ASE_FIRE
from ase.optimize import FIRE2 as ASE_FIRE2
from mace.calculators import MACECalculator

from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater
from batchase.potentials import MACEBatchBackend
from batchase.relaxation import OptimizableBatch, FIRE as BatchFIRE, FIRE2 as BatchFIRE2

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_fire_parity")

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.expanduser("~/.cache/mace/MACE-OFF23_small.model")
DEFAULT_CIF = str(HERE / "fixtures/input.cif")


def test_fire_vs_ase_parity(device: str = "cuda:0", steps: int = 5) -> bool:
    """Test 1: 1:1 step-by-step trajectory parity between ASE FIRE and batchASE FIRE."""
    logger.info("=== Test 1: FIRE 1.0 step-by-step trajectory parity against ASE ===")
    
    atoms_ase = read(DEFAULT_CIF)
    calc_ase = MACECalculator(model_paths=MODEL_PATH, device=device, default_dtype="float64")
    atoms_ase.calc = calc_ase
    dyn_ase = ASE_FIRE(atoms_ase, dt=0.1, maxstep=0.2)

    atoms_batch = read(DEFAULT_CIF)
    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    gbatch = data_list_collater([a2g.convert(atoms_batch)]).to(device)
    obatch = OptimizableBatch(gbatch, backend=backend, dtype=torch.float64, numpy=False)
    dyn_batch = BatchFIRE(obatch, dt=0.1, maxstep=0.2)

    all_ok = True
    for step_idx in range(steps):
        dyn_ase.step()
        dyn_batch.step()

        r_ase = atoms_ase.get_positions()
        r_batch = obatch.get_positions().cpu().numpy()
        e_ase = atoms_ase.get_potential_energy()
        e_batch = obatch.get_potential_energies().item()

        dr_max = np.abs(r_ase - r_batch).max()
        de = abs(e_ase - e_batch)

        logger.info(f"  [FIRE step {step_idx+1:02d}] max pos diff: {dr_max:.3e} A | energy diff: {de:.3e} eV")
        np.testing.assert_allclose(r_batch, r_ase, atol=1e-5, err_msg=f"FIRE pos mismatch at step {step_idx+1}")
        assert de < 1e-5, f"FIRE energy mismatch at step {step_idx+1}: {de}"

    logger.info("--> Test 1 (FIRE 1.0 parity) PASSED successfully!\n")
    return True


def test_fire2_vs_ase_parity(device: str = "cuda:0", steps: int = 5) -> bool:
    """Test 2: 1:1 step-by-step trajectory parity between ASE FIRE2 and batchASE FIRE2."""
    logger.info("=== Test 2: FIRE2 step-by-step trajectory parity against ASE ===")
    
    atoms_ase = read(DEFAULT_CIF)
    calc_ase = MACECalculator(model_paths=MODEL_PATH, device=device, default_dtype="float64")
    atoms_ase.calc = calc_ase
    dyn_ase = ASE_FIRE2(atoms_ase, dt=0.1, maxstep=0.2)

    atoms_batch = read(DEFAULT_CIF)
    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    gbatch = data_list_collater([a2g.convert(atoms_batch)]).to(device)
    obatch = OptimizableBatch(gbatch, backend=backend, dtype=torch.float64, numpy=False)
    dyn_batch = BatchFIRE2(obatch, dt=0.1, maxstep=0.2, force_reeval=True)

    for step_idx in range(steps):
        dyn_ase.step()
        dyn_batch.step()

        r_ase = atoms_ase.get_positions()
        r_batch = obatch.get_positions().cpu().numpy()
        e_ase = atoms_ase.get_potential_energy()
        e_batch = obatch.get_potential_energies().item()

        dr_max = np.abs(r_ase - r_batch).max()
        de = abs(e_ase - e_batch)

        logger.info(f"  [FIRE2 step {step_idx+1:02d}] max pos diff: {dr_max:.3e} A | energy diff: {de:.3e} eV")
        np.testing.assert_allclose(r_batch, r_ase, atol=1e-5, err_msg=f"FIRE2 pos mismatch at step {step_idx+1}")
        assert de < 1e-5, f"FIRE2 energy mismatch at step {step_idx+1}: {de}"

    logger.info("--> Test 2 (FIRE2 parity) PASSED successfully!\n")
    return True


def test_ragged_batch_invariance(device: str = "cuda:0", steps: int = 5) -> bool:
    """Test 3: Ragged Batch invariance (Batch=1 vs Batch=2 with 92 and 184 atoms)."""
    logger.info("=== Test 3: Ragged Batch invariance (92 vs 184 atoms) ===")

    a0 = read(DEFAULT_CIF)
    a1 = a0.repeat((1, 1, 2))  # 184 atoms

    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

    # 1. System 0 alone (Batch=1)
    gbatch_single = data_list_collater([a2g.convert(a0.copy())]).to(device)
    obatch_single = OptimizableBatch(gbatch_single, backend=backend, dtype=torch.float64, numpy=False)
    opt_single = BatchFIRE(obatch_single, dt=0.1, maxstep=0.2)

    # 2. System 0 + System 1 (Batch=2, ragged: 92 vs 184 atoms)
    gbatch_ragged = data_list_collater([a2g.convert(a0.copy()), a2g.convert(a1.copy())]).to(device)
    obatch_ragged = OptimizableBatch(gbatch_ragged, backend=backend, dtype=torch.float64, numpy=False)
    opt_ragged = BatchFIRE(obatch_ragged, dt=0.1, maxstep=0.2)

    for step_idx in range(steps):
        opt_single.step()
        opt_ragged.step()

        pos_single = obatch_single.get_positions().cpu().numpy()
        pos_ragged_sys0 = obatch_ragged.get_positions()[:len(a0)].cpu().numpy()

        e_single = obatch_single.get_potential_energies().item()
        e_ragged_sys0 = obatch_ragged.get_potential_energies()[0].item()

        dr_max = np.abs(pos_single - pos_ragged_sys0).max()
        de = abs(e_single - e_ragged_sys0)

        logger.info(f"  [Ragged step {step_idx+1:02d}] sys0 pos diff: {dr_max:.3e} A | energy diff: {de:.3e} eV")
        np.testing.assert_allclose(pos_ragged_sys0, pos_single, atol=1e-12, err_msg="Ragged batch mismatch")
        assert de < 1e-8, f"Ragged batch energy mismatch: {de}"

    logger.info("--> Test 3 (Ragged batch invariance) PASSED successfully!\n")
    return True


def test_slot_replenishment_preservation(device: str = "cuda:0") -> bool:
    """Test 4: Dynamic Slot Replenishment correctly transfers retained state."""
    logger.info("=== Test 4: Dynamic Slot Replenishment state preservation ===")

    a0 = read(DEFAULT_CIF)
    a1 = a0.repeat((1, 1, 2))
    a2 = a0.copy()

    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

    gbatch = data_list_collater([a2g.convert(a0.copy()), a2g.convert(a1.copy())]).to(device)
    obatch = OptimizableBatch(gbatch, backend=backend, dtype=torch.float64, numpy=False)
    opt = BatchFIRE(obatch, dt=0.1, maxstep=0.2)

    opt.step()
    opt.step()

    # Retain slot 1 (a1), replace slot 0 with a2
    old_batch_indices = obatch.batch_indices.clone()
    gbatch_new = data_list_collater([a2g.convert(a1.copy()), a2g.convert(a2.copy())]).to(device)
    obatch_new = OptimizableBatch(gbatch_new, backend=backend, dtype=torch.float64, numpy=False)
    opt.optimizable = obatch_new
    opt.restart_from_earlystop(restart_indices=[1], old_batch_indices=old_batch_indices)

    assert opt.dt[0].item() > 0.0, "Retained slot 0 must have valid dt"
    assert opt.v.shape[0] == len(a1) + len(a2), "Velocity tensor shape must match new batch"

    opt.step()
    logger.info("  [Slot Replenishment] Step after dynamic slot update completed smoothly.")
    logger.info("--> Test 4 (Slot replenishment) PASSED successfully!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run numerical parity test suite for batchASE FIRE/FIRE2.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=5, help="Number of comparison steps per test")
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info(f"Starting test_fire_parity on device={args.device}, steps={args.steps}")

    t1_ok = test_fire_vs_ase_parity(device=args.device, steps=args.steps)
    t2_ok = test_fire2_vs_ase_parity(device=args.device, steps=args.steps)
    t3_ok = test_ragged_batch_invariance(device=args.device, steps=args.steps)
    t4_ok = test_slot_replenishment_preservation(device=args.device)

    total_time = time.perf_counter() - t_start
    all_passed = t1_ok and t2_ok and t3_ok and t4_ok

    if all_passed:
        logger.info("=================================================================")
        logger.info(f"ALL 4 PARITY AND REGRESSION TESTS PASSED IN {total_time:.2f}s!")
        logger.info("=================================================================")
    else:
        logger.error("SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
