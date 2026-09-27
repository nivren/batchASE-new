from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from batchase.engine.worker import Worker
from batchase.relaxation.optimizers.bfgsfusedls import BFGSFusedLS
from batchase.relaxation.optimizers.lbfgs import LBFGS


class _FinishedOptimizable:
    def __init__(self):
        self.device = torch.device("cpu")
        self.batch_indices = torch.tensor([0], dtype=torch.long)
        self.results = {}
        self.batch = SimpleNamespace()
        self.converge_indices_list = []
        self.failed_indices_list = []

    @property
    def batch_size(self):
        return 1

    def get_max_forces(self, *args, **kwargs):
        return torch.zeros(1, dtype=torch.float64)

    def converged(self, *args, **kwargs):
        return True


class TestBurstCacheBehavior(unittest.TestCase):
    def test_optimizers_do_not_flush_cache_per_run(self):
        optimizable = _FinishedOptimizable()

        with patch("batchase.relaxation.optimizers.lbfgs.torch.cuda.empty_cache") as lbfgs_cache:
            LBFGS(optimizable, early_stop=False).run(fmax=0.01, steps=5)
            lbfgs_cache.assert_not_called()

        fused = BFGSFusedLS(optimizable, early_stop=False, device="cpu")
        with patch("batchase.relaxation.optimizers.bfgsfusedls.torch.cuda.empty_cache") as fused_cache, patch(
            "gc.collect"
        ) as gc_collect:
            fused.run(fmax=0.01, steps=5)
            fused_cache.assert_not_called()
            gc_collect.assert_not_called()

    def test_stage_memory_cleanup_is_device_aware(self):
        worker = Worker.__new__(Worker)
        worker.device = "cuda:0"
        with patch("batchase.engine.worker.gc.collect") as gc_collect, patch(
            "batchase.engine.worker.torch.cuda.empty_cache"
        ) as empty_cache:
            worker._clear_stage_memory()
            gc_collect.assert_called_once_with()
            empty_cache.assert_called_once_with()

        worker.device = "cpu"
        with patch("batchase.engine.worker.gc.collect") as gc_collect, patch(
            "batchase.engine.worker.torch.cuda.empty_cache"
        ) as empty_cache:
            worker._clear_stage_memory()
            gc_collect.assert_called_once_with()
            empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
