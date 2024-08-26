import unittest

import torch
import torch.nn as nn

from fusion_bench.method.simple_average import SimpleAverageAlgorithm


class TestSimpleAverageAlgorithm(unittest.TestCase):
    def setUp(self):
        self.algorithm = SimpleAverageAlgorithm()
        self.models = [nn.Linear(10, 1) for _ in range(5)]

    def test_run(self):
        merged_model = self.algorithm.run(self.models)
        self.assertIsInstance(merged_model, nn.Module)

        # Check that the parameters of the merged model are the average of the parameters of the input models
        for name, param in merged_model.named_parameters():
            self.assertTrue(
                torch.allclose(
                    param,
                    sum(model.state_dict()[name] for model in self.models)
                    / len(self.models),
                    atol=1e-6,
                )
            )

    def test_run_with_dict(self):
        model_dict = {f"model_{i}": model for i, model in enumerate(self.models)}
        merged_model = self.algorithm.run(model_dict)
        self.assertIsInstance(merged_model, nn.Module)

        # Check that the parameters of the merged model are the average of the parameters of the input models
        for name, param in merged_model.named_parameters():
            self.assertTrue(
                torch.allclose(
                    param,
                    sum(model.state_dict()[name] for model in model_dict.values())
                    / len(model_dict),
                    atol=1e-6,
                )
            )


if __name__ == "__main__":
    unittest.main()
