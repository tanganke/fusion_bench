"""Tests for the SWUDI / ASWUDI model-merging algorithm."""

import math
import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from fusion_bench.method.swudi import SWUDIMerging, swudi_merging


def _normal_equation(task_vectors):
    """C and D of the WUDI normal equation ``tau C = D``, in float64."""
    tvs = task_vectors.double()
    weights = 1.0 / tvs.flatten(1).square().sum(dim=1)
    gram = tvs.mT @ tvs
    C = torch.einsum("n,nij->ij", weights, gram)
    D = torch.einsum("n,nij->ij", weights, tvs @ gram)
    return C, D


def _merge_layer(task_vectors, **kwargs):
    merged = swudi_merging(
        [{"w": tv} for tv in task_vectors], accelerator="cpu", **kwargs
    )
    return merged["w"].double()


class TestSWUDIMergingFunction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Decaying column scales give C a spectrum with a clear tail.
        scales = torch.tensor([3.0, 2.0, 1.0, 0.3, 0.1, 0.05])
        self.tvs = torch.randn(3, 8, 6) * scales  # (num_tasks, d_out, d_in)
        self.C, self.D = _normal_equation(self.tvs)
        self.tau_init = self.tvs.double().sum(dim=0)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def assert_truncated_inverse(self, merged, rank):
        """Top-``rank`` directions of C are inverted exactly, the others stay at ``tau_init``."""
        eigvals, Q = torch.linalg.eigh(self.C)
        kept, dropped = Q[:, -rank:], Q[:, :-rank]
        self.assert_close(merged @ kept, self.D @ kept / eigvals[-rank:])
        self.assert_close(merged @ dropped, self.tau_init @ dropped)

    def test_zero_time_returns_initial_point(self):
        merged = _merge_layer(self.tvs, exp_time=0.0, rank_ratio=1.0)
        self.assert_close(merged, self.tau_init)

    def test_soft_filter_matches_gradient_flow(self):
        # Gradient flow d(tau)/dt = D - tau C started at tau_init.
        t = 2.0
        tau_cf = torch.linalg.solve(self.C, self.D.T).T
        expected = tau_cf + (self.tau_init - tau_cf) @ torch.linalg.matrix_exp(
            -t * self.C
        )
        merged = _merge_layer(self.tvs, exp_time=t, rank_ratio=1.0)
        self.assert_close(merged, expected)

    def test_hard_truncation(self):
        merged = _merge_layer(self.tvs, exp_time=None, rank_ratio=0.5)
        self.assert_truncated_inverse(merged, rank=3)

    def test_aswudi_uses_participation_ratio_rank(self):
        eigvals = torch.linalg.eigvalsh(self.C)
        rank = math.ceil(eigvals.sqrt().sum() ** 2 / eigvals.sum())
        self.assertTrue(1 < rank < self.C.shape[0])
        merged = _merge_layer(self.tvs, exp_time=None, rank_ratio=None)
        self.assert_truncated_inverse(merged, rank)

    def test_non_linear_and_excluded_parameters_are_averaged(self):
        task_vectors = [{"w": tv, "b": torch.randn(8)} for tv in self.tvs]
        merged = swudi_merging(task_vectors, exclude_keys=["w"], accelerator="cpu")
        for key in ("w", "b"):
            expected = torch.stack([tv[key] for tv in task_vectors]).mean(dim=0)
            torch.testing.assert_close(merged[key], expected)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 8)
        self.norm = nn.LayerNorm(8)


class TestSWUDIMergingAlgorithm(unittest.TestCase):
    def test_run_against_dict_modelpool(self):
        torch.manual_seed(0)
        pretrained = TinyModel()
        models = {"_pretrained_": deepcopy(pretrained)}
        for i in range(3):
            finetuned = deepcopy(pretrained)
            with torch.no_grad():
                for p in finetuned.parameters():
                    p.add_(torch.randn_like(p) * 0.1)
            models[f"task_{i}"] = finetuned

        algorithm = SWUDIMerging(exp_time=None, rank_ratio=None, scaling_factor=0.5)
        merged = algorithm.run(models)

        pretrained_sd = pretrained.state_dict()
        task_vectors = [
            {
                k: v - pretrained_sd[k]
                for k, v in models[f"task_{i}"].state_dict().items()
            }
            for i in range(3)
        ]
        merged_tv = swudi_merging(
            task_vectors,
            exp_time=None,
            rank_ratio=None,
            accelerator=algorithm.fabric.device,
        )
        for k, v in merged.state_dict().items():
            torch.testing.assert_close(v, pretrained_sd[k] + 0.5 * merged_tv[k])


if __name__ == "__main__":
    unittest.main()
