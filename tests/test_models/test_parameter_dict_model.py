import unittest

import torch
from torch import nn

from fusion_bench.models.parameter_dict import ParameterDictModel


class TestParameterDictModel(unittest.TestCase):
    def setUp(self):
        # Create a simple parameter dictionary
        self.parameters = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(20, 10),
            "layer2.bias": torch.randn(20),
        }
        for name, param in self.parameters.items():
            self.parameters[name] = nn.Parameter(param, requires_grad=False)
        self.model = ParameterDictModel(self.parameters)

    def test_repr(self):
        # Test the __repr__ method
        representation = repr(self.model)
        self.assertIsInstance(representation, str)
        self.assertTrue(len(representation) > 0)
        print(self.model)

    def test_parameters_equal(self):
        # Test the parameters
        for name, param in self.parameters.items():
            self.assertTrue(
                torch.all((self.model.get_parameter(name) - param) == 0),
                f"Failed for {name}",
            )


if __name__ == "__main__":
    unittest.main()
