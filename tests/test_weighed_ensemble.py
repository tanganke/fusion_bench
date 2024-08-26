import unittest

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from fusion_bench.method.ensemble import WeightedEnsembleAlgorithm


class TestWeightedAlgorithm(unittest.TestCase):
    def setUp(self):
        self.method_config = {"name": "weighted_ensemble", "weights": [0.3, 0.7]}
        self.algorithm = WeightedEnsembleAlgorithm(DictConfig(self.method_config))

        # Assume we have a list of PyTorch models (nn.Module instances) that we want to ensemble.
        # Replace with actual models
        self.models = [nn.Linear(10, 1) for _ in range(2)]

    def test_run(self):
        merged_model = self.algorithm.run(self.models)
        self.assertIsNotNone(merged_model)

        outputs = [merged_model(torch.randn(10)) for _ in range(10)]
        self.assertTrue(
            all(torch.is_tensor(output) and output.dim() == 1 for output in outputs)
        )


if __name__ == "__main__":
    unittest.main()
