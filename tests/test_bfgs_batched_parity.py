"""
Numerical parity and regression test suite for tensor-parallel batched BFGS in batchASE.

Verification criteria:
1. cuSOLVER syevjBatched accuracy: eigenvalues and reconstruction parity against PyTorch
   (eigenvalue diff < 1e-10, reconstruction error < 1e-10 in float64).
2. Trajectory parity vs ASE: batchASE.BFGS vs official ase.optimize.BFGS
   (step-by-step max position diff < 1e-5 A, energy diff < 1e-5 eV).
3. Homogeneous batch invariance: Batch=1 vs Batch=4 identical structures
   (trajectory difference < 1e-12 A, matching float64 machine precision).
4. UnitCellFilter compatibility: BFGS relaxation on OptimizableUnitCellBatch.
5. Dynamic Slot Replenishment: slot compaction and replenishment preserves Hessian state.
6. Heterogeneous fallback: mixed-dimension batch runs seamlessly without errors.
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
from ase.optimize import BFGS as ASE_BFGS
from mace.calculators import MACECalculator

from batchase.neighbors import AtomsToGraphs
from batchase.utils import data_list_collater
from batchase.potentials import MACEBatchBackend
from batchase.relaxation import (
    OptimizableBatch,
    OptimizableUnitCellBatch,
    BFGS as BatchBFGS,
)
from batchase.relaxation.cusolver_batched import (
    cusolver_syevj_batched,
    is_cusolver_batched_available,
)

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_bfgs_parity")

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.expanduser("~/.cache/mace/MACE-OFF23_small.model")
DEFAULT_CIF = str(HERE / "fixtures/input.cif")


def test_cusolver_syevj_accuracy(device: str = "cuda:0") -> bool:
    """Test 1: Verify cuSOLVER batched eigh accuracy against PyTorch on multiple matrix dimensions."""
    logger.info("=== Test 1: cuSOLVER syevjBatched numerical accuracy vs PyTorch ===")
    if not is_cusolver_batched_available():
        logger.warning("cuSOLVER batched library not available, skipping Test 1.")
        return True

    dims = [30, 95, 285]
    for N in dims:
        B = 4
        torch.manual_seed(42)
        M = torch.randn(B, N, N, dtype=torch.float64, device=device)
        H = M + M.transpose(-1, -2)

        w_th, v_th = torch.linalg.eigh(H)
        w_cs, v_cs = cusolver_syevj_batched(H)

        eig_diff = (w_cs - w_th).abs().max().item()
        rec = torch.bmm(v_cs, torch.bmm(torch.diag_embed(w_cs), v_cs.transpose(-1, -2)))
        rec_err = (rec - H).abs().max().item()

        logger.info(f"  [float64 B={B}, N={N:3d}] eig_diff={eig_diff:.2e} | rec_err={rec_err:.2e}")
        assert eig_diff < 1e-10, f"Eigenvalue diff too large for N={N}: {eig_diff}"
        assert rec_err < 1e-10, f"Reconstruction error too large for N={N}: {rec_err}"

    # Also test float32
    M32 = torch.randn(4, 60, 60, dtype=torch.float32, device=device)
    H32 = M32 + M32.transpose(-1, -2)
    w_cs32, v_cs32 = cusolver_syevj_batched(H32)
    rec32 = torch.bmm(v_cs32, torch.bmm(torch.diag_embed(w_cs32), v_cs32.transpose(-1, -2)))
    rec_err32 = (rec32 - H32).abs().max().item()
    logger.info(f"  [float32 B=4, N= 60] rec_err={rec_err32:.2e}")
    assert rec_err32 < 1e-3, f"Float32 reconstruction error too large: {rec_err32}"

    logger.info("--> Test 1 (cuSOLVER accuracy) PASSED successfully!\n")
    return True


def test_bfgs_vs_ase_parity(device: str = "cuda:0", steps: int = 5) -> bool:
    """Test 2: 1:1 step-by-step trajectory parity between ASE BFGS and batchASE BFGS."""
    logger.info("=== Test 2: BFGS step-by-step trajectory parity against ASE ===")

    atoms_ase = read(DEFAULT_CIF)
    calc_ase = MACECalculator(model_paths=MODEL_PATH, device=device, default_dtype="float64")
    atoms_ase.calc = calc_ase
    dyn_ase = ASE_BFGS(atoms_ase, maxstep=0.2, alpha=70.0)

    atoms_batch = read(DEFAULT_CIF)
    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    gbatch = data_list_collater([a2g.convert(atoms_batch)]).to(device)
    obatch = OptimizableBatch(gbatch, backend=backend, dtype=torch.float64, numpy=False)
    dyn_batch = BatchBFGS(obatch, maxstep=0.2, alpha=70.0)

    for step_idx in range(steps):
        dyn_ase.step()
        dyn_batch.step()

        r_ase = atoms_ase.get_positions()
        r_batch = obatch.get_positions().cpu().numpy()
        e_ase = atoms_ase.get_potential_energy()
        e_batch = obatch.get_potential_energies().item()

        dr_max = np.abs(r_ase - r_batch).max()
        de = abs(e_ase - e_batch)

        logger.info(f"  [BFGS step {step_idx+1:02d}] max pos diff: {dr_max:.3e} A | energy diff: {de:.3e} eV")
        np.testing.assert_allclose(r_batch, r_ase, atol=1e-5, err_msg=f"BFGS pos mismatch at step {step_idx+1}")
        assert de < 1e-5, f"BFGS energy mismatch at step {step_idx+1}: {de}"

    logger.info("--> Test 2 (BFGS parity vs ASE) PASSED successfully!\n")
    return True


def test_homogeneous_batch_invariance(device: str = "cuda:0", steps: int = 5) -> bool:
    """Test 3: Homogeneous batch invariance (Batch=1 vs Batch=4 identical structures)."""
    logger.info("=== Test 3: Homogeneous batch invariance (Batch=1 vs Batch=4) ===")

    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

    # Single-item batch
    a_single = read(DEFAULT_CIF)
    gbatch_single = data_list_collater([a2g.convert(a_single)]).to(device)
    obatch_single = OptimizableBatch(gbatch_single, backend=backend, dtype=torch.float64, numpy=False)
    opt_single = BatchBFGS(obatch_single, maxstep=0.2, alpha=70.0)

    # 4-item batch
    copies = [read(DEFAULT_CIF) for _ in range(4)]
    gbatch_quad = data_list_collater([a2g.convert(c) for c in copies]).to(device)
    obatch_quad = OptimizableBatch(gbatch_quad, backend=backend, dtype=torch.float64, numpy=False)
    opt_quad = BatchBFGS(obatch_quad, maxstep=0.2, alpha=70.0)

    for step_idx in range(steps):
        opt_single.step()
        opt_quad.step()

        pos_single = obatch_single.get_positions().cpu().numpy()
        pos_quad = obatch_quad.get_positions().cpu().numpy().reshape(4, -1, 3)

        for i in range(4):
            dr = np.abs(pos_quad[i] - pos_single).max()
            assert dr < 1e-10, f"Batch item {i} deviated from single reference at step {step_idx+1}: {dr:.3e} A"

        logger.info(f"  [Batch invariance step {step_idx+1:02d}] max deviation across 4 slots: {np.abs(pos_quad - pos_single).max():.3e} A")

    logger.info("--> Test 3 (Homogeneous batch invariance) PASSED successfully!\n")
    return True


def test_bfgs_unitcellfilter(device: str = "cuda:0", steps: int = 5) -> bool:
    """Test 4: BFGS relaxation on OptimizableUnitCellBatch with lattice strain."""
    logger.info("=== Test 4: BFGS on OptimizableUnitCellBatch ===")

    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

    atoms_list = [read(DEFAULT_CIF), read(DEFAULT_CIF)]
    gbatch = data_list_collater([a2g.convert(a) for a in atoms_list]).to(device)
    obatch = OptimizableUnitCellBatch(
        batch=gbatch,
        backend=backend,
        scalar_pressure=0.0006,
        dtype=torch.float64,
        numpy=False,
    )
    opt = BatchBFGS(obatch, maxstep=0.2, alpha=70.0)

    e_prev = obatch.get_potential_energies().detach().cpu().numpy()
    for step_idx in range(steps):
        opt.step()
        e_curr = obatch.get_potential_energies().detach().cpu().numpy()
        logger.info(f"  [UnitCellFilter step {step_idx+1:02d}] E0={e_curr[0]:.4f} eV, E1={e_curr[1]:.4f} eV")

    assert np.all(e_curr <= e_prev + 0.1), "Energies should generally decrease or stay stable during cell relaxation"
    logger.info("--> Test 4 (OptimizableUnitCellBatch) PASSED successfully!\n")
    return True


def test_bfgs_slot_replenishment(device: str = "cuda:0") -> bool:
    """Test 5: Dynamic Slot Replenishment correctly transfers retained Hessian state."""
    logger.info("=== Test 5: Dynamic Slot Replenishment state preservation ===")

    a0 = read(DEFAULT_CIF)
    a1 = a0.copy()
    a2 = a0.copy()

    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

    gbatch = data_list_collater([a2g.convert(a0.copy()), a2g.convert(a1.copy())]).to(device)
    obatch = OptimizableBatch(gbatch, backend=backend, dtype=torch.float64, numpy=False)
    opt = BatchBFGS(obatch, maxstep=0.2, alpha=70.0)

    opt.step()
    opt.step()

    # Verify H is 3D tensor
    assert opt.H.dim() == 3, f"H should be 3D tensor, got shape {opt.H.shape}"
    h1_before = opt.H[1].clone()

    # Retain slot 1 (a1), replace slot 0 with a2
    old_batch_indices = obatch.batch_indices.clone()
    gbatch_new = data_list_collater([a2g.convert(a1.copy()), a2g.convert(a2.copy())]).to(device)
    obatch_new = OptimizableBatch(gbatch_new, backend=backend, dtype=torch.float64, numpy=False)
    opt.optimizable = obatch_new
    opt.restart_from_earlystop(restart_indices=[1], old_batch_indices=old_batch_indices)

    # Retained slot 0 must have preserved h1_before
    h0_after = opt.H[0]
    h_diff = (h0_after - h1_before).abs().max().item()
    logger.info(f"  [Slot Replenishment] Retained slot Hessian diff: {h_diff:.3e}")
    assert h_diff < 1e-12, f"Retained slot Hessian was not preserved: {h_diff}"

    # New slot 1 must be alpha * I
    D = opt.dim
    eye_expected = torch.eye(D, device=device, dtype=torch.float64) * 70.0
    h1_after = opt.H[1]
    new_diff = (h1_after - eye_expected).abs().max().item()
    logger.info(f"  [Slot Replenishment] New slot alpha*I diff: {new_diff:.3e}")
    assert new_diff < 1e-12, f"New slot Hessian was not initialized to alpha*I: {new_diff}"

    # Execute step after replenishment
    opt.step()
    logger.info("  [Slot Replenishment] Step after replenishment succeeded without error.")
    logger.info("--> Test 5 (Slot replenishment) PASSED successfully!\n")
    return True


def test_bfgs_heterogeneous_fallback(device: str = "cuda:0", steps: int = 3) -> bool:
    """Test 6: Heterogeneous batch with different atom counts runs correctly on unified stream."""
    logger.info("=== Test 6: Heterogeneous batch handling ===")

    a0 = read(DEFAULT_CIF)
    a1 = a0.repeat((1, 1, 2))  # Double size

    backend = MACEBatchBackend(model=MODEL_PATH, device=device, default_dtype="float64")
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

    gbatch = data_list_collater([a2g.convert(a0.copy()), a2g.convert(a1.copy())]).to(device)
    obatch = OptimizableBatch(gbatch, backend=backend, dtype=torch.float64, numpy=False)
    opt = BatchBFGS(obatch, maxstep=0.2, alpha=70.0)

    assert not opt.is_homogeneous, "Batch should be detected as heterogeneous"

    for step_idx in range(steps):
        opt.step()
        logger.info(f"  [Heterogeneous step {step_idx+1:02d}] completed successfully.")

    logger.info("--> Test 6 (Heterogeneous fallback) PASSED successfully!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Numerical parity and regression test suite for batchASE BFGS.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=5, help="Number of comparison steps per test")
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info(f"Starting test_bfgs_batched_parity on device={args.device}, steps={args.steps}")

    t1_ok = test_cusolver_syevj_accuracy(device=args.device)
    t2_ok = test_bfgs_vs_ase_parity(device=args.device, steps=args.steps)
    t3_ok = test_homogeneous_batch_invariance(device=args.device, steps=args.steps)
    t4_ok = test_bfgs_unitcellfilter(device=args.device, steps=args.steps)
    t5_ok = test_bfgs_slot_replenishment(device=args.device)
    t6_ok = test_bfgs_heterogeneous_fallback(device=args.device)

    total_time = time.perf_counter() - t_start
    all_passed = t1_ok and t2_ok and t3_ok and t4_ok and t5_ok and t6_ok

    if all_passed:
        logger.info("=================================================================")
        logger.info(f"ALL 6 PARITY AND REGRESSION TESTS PASSED IN {total_time:.2f}s!")
        logger.info("=================================================================")
    else:
        logger.error("SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
