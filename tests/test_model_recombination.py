import unittest
from unittest.mock import MagicMock, patch

from torch import nn

from fusion_bench.method import ModelRecombinationAlgorithm
from fusion_bench.modelpool import ModelPool, to_modelpool


class TestModelRecombinationAlgorithm(unittest.TestCase):
    def setUp(self):
        self.model_recombination = ModelRecombinationAlgorithm()

    def test_run_shuffle_state_dict(self):
        models = [nn.Linear(10, 10) for _ in range(3)]
        modelpool = to_modelpool(models)
        new_modelpool = self.model_recombination.run(modelpool, return_modelpool=True)
        self.assertIsInstance(new_modelpool, ModelPool)

        new_model = self.model_recombination.run(modelpool, return_modelpool=False)
        self.assertIsInstance(new_model, nn.Module)

    def test_run_shuffle_state_dict_with_dict(self):
        models = [
            nn.ModuleDict({f"layer_{i}": nn.Linear(10, 10) for i in range(3)})
            for _ in range(3)
        ]
        modelpool = to_modelpool(models)
        new_modelpool = self.model_recombination.run(modelpool, return_modelpool=True)
        self.assertIsInstance(new_modelpool, ModelPool)
        self.assertEqual(len(new_modelpool), 3)
        self.assertIsInstance(
            new_modelpool.load_model(new_modelpool.model_names[0]), nn.ModuleDict
        )

        new_model = self.model_recombination.run(modelpool, return_modelpool=False)
        self.assertIsInstance(new_model, nn.Module)

    def test_run_shuffle_state_dict_with_list(self):
        models = [
            nn.ModuleList([nn.Linear(10, 10) for _ in range(3)]) for _ in range(3)
        ]
        modelpool = to_modelpool(models)
        new_modelpool = self.model_recombination.run(modelpool, return_modelpool=True)
        self.assertIsInstance(new_modelpool, ModelPool)
        self.assertEqual(len(new_modelpool), 3)
        self.assertIsInstance(
            new_modelpool.load_model(new_modelpool.model_names[0]), nn.ModuleList
        )

        new_model = self.model_recombination.run(modelpool, return_modelpool=False)
        self.assertIsInstance(new_model, nn.Module)


if __name__ == "__main__":
    unittest.main()
