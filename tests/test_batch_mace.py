from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from ase.build import molecule

from batchase.neighbors import AtomsToGraphs
from batchase.potentials import create_backend
from batchase.utils import data_list_collater


MODEL_PATH = Path.home() / ".cache/mace/MACE-OFF23_small.model"


@unittest.skipUnless(torch.cuda.is_available() and MODEL_PATH.is_file(), "CUDA or cached MACE unavailable")
class TestBatchMACE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = create_backend(f"mace_off:{MODEL_PATH}", device="cuda:0")
        cls.atoms = []
        for name in ("H2O", "CH4"):
            atoms = molecule(name)
            atoms.set_cell([12.0] * 3)
            atoms.center()
            atoms.pbc = True
            cls.atoms.append(atoms)

    def test_same_backend_handles_successive_different_sized_batches(self):
        for atoms_list in (self.atoms, self.atoms[:1], self.atoms[::-1]):
            results = self.backend.predict_from_atoms(atoms_list, compute_stress=True)
            offset = 0
            for index, atoms in enumerate(atoms_list):
                reference = atoms.copy()
                reference.calc = self.backend.calculator
                np.testing.assert_allclose(results["energy"][index].cpu().numpy(),
                                           reference.get_potential_energy(), rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(results["forces"][offset:offset + len(atoms)].cpu().numpy(),
                                           reference.get_forces(), rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(results["stress"][index].cpu().numpy(),
                                           reference.get_stress(voigt=False), rtol=1e-8, atol=1e-8)
                offset += len(atoms)

    def test_cuda_neighbors_never_connect_different_structures(self):
        a2g = AtomsToGraphs(r_edges=False, r_pbc=True, dtype=torch.float64)
        graph = data_list_collater([a2g.convert(atoms) for atoms in self.atoms]).to("cuda:0")
        inputs = self.backend.build_inputs(graph)
        edge_index = inputs["edge_index"]
        self.assertGreater(edge_index.shape[1], 0)
        torch.testing.assert_close(inputs["batch"][edge_index[0]], inputs["batch"][edge_index[1]])
        self.assertEqual(inputs["ptr"].tolist(), [0, 3, 8])


if __name__ == "__main__":
    unittest.main()
