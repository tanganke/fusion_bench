"""Tests for the MagMax model-merging algorithm.

Cross-checked against the reference implementation
(``tmp/magmax/src/merging/task_vectors.py::merge_max_abs``).
"""

import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from fusion_bench.method.magmax import MagMaxAlgorithm, magmax_merge
from fusion_bench.method.magmax.magmax import _magmax_select


class TinyModel(nn.Module):
    def __init__(self, in_dim=4, out_dim=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)


def _reference_merge_max_abs(task_vectors):
    """Direct port of ``merge_max_abs`` from the reference repo, used as a
    ground-truth oracle for the FusionBench implementation."""
    new_vector = {}
    for key in task_vectors[0]:
        max_abs_tensor = task_vectors[0][key].clone()
        for tv in task_vectors[1:]:
            current = tv[key]
            max_abs_tensor = torch.where(
                current.abs() >= max_abs_tensor.abs(), current, max_abs_tensor
            )
        new_vector[key] = max_abs_tensor
    return new_vector


class TestMagMaxSelect(unittest.TestCase):
    def test_matches_reference_implementation(self):
        torch.manual_seed(0)
        tvs = [
            {"w": torch.randn(5, 7), "b": torch.randn(5)}
            for _ in range(4)
        ]
        oracle = _reference_merge_max_abs(tvs)
        ours = _magmax_select(tvs)
        for k in oracle:
            self.assertTrue(torch.equal(oracle[k], ours[k]), msg=f"mismatch on key {k}")

    def test_per_element_max_magnitude(self):
        tv_a = {"w": torch.tensor([1.0, -2.0, 3.0])}
        tv_b = {"w": torch.tensor([-4.0, 0.5, 2.0])}
        tv_c = {"w": torch.tensor([0.0, -2.5, 3.0])}
        out = _magmax_select([tv_a, tv_b, tv_c])
        # max-abs values per index are: 4.0 (from b), 2.5 (from c), 3.0 (a or c)
        expected = torch.tensor([-4.0, -2.5, 3.0])
        self.assertTrue(torch.equal(out["w"], expected))

    def test_tie_breaking_prefers_later(self):
        # On ties (|a| == |b|), the reference uses '>=' so the LATER task wins.
        tv_a = {"w": torch.tensor([1.0])}
        tv_b = {"w": torch.tensor([-1.0])}
        out = _magmax_select([tv_a, tv_b])
        self.assertTrue(torch.equal(out["w"], torch.tensor([-1.0])))


class TestMagMaxMerge(unittest.TestCase):
    def test_merge_recovers_pretrained_plus_scaled_taumax(self):
        torch.manual_seed(42)
        pretrained = TinyModel()
        finetuned = [deepcopy(pretrained) for _ in range(3)]
        # Make each finetuned model differ from pretrained
        for ft in finetuned:
            with torch.no_grad():
                for p in ft.parameters():
                    p.add_(torch.randn_like(p) * 0.1)

        scaling = 0.5
        merged = magmax_merge(
            deepcopy(pretrained), finetuned, scaling_factor=scaling, inplace=True
        )

        # Recompute expected merge via the reference path
        ptm_sd = pretrained.state_dict()
        tvs = [
            {k: ft.state_dict()[k] - ptm_sd[k] for k in ptm_sd}
            for ft in finetuned
        ]
        tau_max = _reference_merge_max_abs(tvs)
        expected_sd = {k: ptm_sd[k] + scaling * tau_max[k] for k in ptm_sd}

        for k, v in merged.state_dict().items():
            self.assertTrue(
                torch.allclose(v, expected_sd[k]), msg=f"mismatch on key {k}"
            )

    def test_scaling_factor_zero_returns_pretrained(self):
        torch.manual_seed(1)
        pretrained = TinyModel()
        finetuned = [TinyModel() for _ in range(3)]
        original_sd = {k: v.clone() for k, v in pretrained.state_dict().items()}
        merged = magmax_merge(
            deepcopy(pretrained), finetuned, scaling_factor=0.0, inplace=True
        )
        for k, v in merged.state_dict().items():
            self.assertTrue(torch.allclose(v, original_sd[k]))


class TestMagMaxAlgorithm(unittest.TestCase):
    def test_run_against_dict_modelpool(self):
        torch.manual_seed(7)
        pretrained = TinyModel()
        models = {"_pretrained_": deepcopy(pretrained)}
        for i in range(3):
            ft = deepcopy(pretrained)
            with torch.no_grad():
                for p in ft.parameters():
                    p.add_(torch.randn_like(p) * 0.05 * (i + 1))
            models[f"task_{i}"] = ft

        algo = MagMaxAlgorithm(scaling_factor=0.5, inplace=False)
        merged = algo.run(models)

        # Same expected output as the direct function call.
        finetuned = [models[f"task_{i}"] for i in range(3)]
        expected = magmax_merge(
            deepcopy(pretrained), finetuned, scaling_factor=0.5, inplace=False
        )
        for k, v in merged.state_dict().items():
            self.assertTrue(torch.allclose(v, expected.state_dict()[k]))


if __name__ == "__main__":
    unittest.main()
