import unittest

import torch
from torch import nn

from fusion_bench.models.parameter_dict import ParameterDictModel


class TestParameterDictModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with various parameter configurations."""
        # Create a simple parameter dictionary with nested structure
        self.parameters = {
            "layer1.weight": nn.Parameter(torch.randn(10, 10), requires_grad=False),
            "layer1.bias": nn.Parameter(torch.randn(10), requires_grad=False),
            "layer2.weight": nn.Parameter(torch.randn(20, 10), requires_grad=False),
            "layer2.bias": nn.Parameter(torch.randn(20), requires_grad=False),
        }
        self.model = ParameterDictModel(self.parameters)

    def test_initialization_empty(self):
        """Test initialization with no parameters."""
        model = ParameterDictModel()
        self.assertEqual(len(model), 0)
        self.assertEqual(list(model.keys()), [])

    def test_initialization_with_parameters(self):
        """Test initialization with parameters."""
        self.assertEqual(len(self.model), len(self.parameters))
        for name in self.parameters.keys():
            self.assertIn(name, self.model)

    def test_repr(self):
        """Test the __repr__ method."""
        representation = repr(self.model)
        self.assertIsInstance(representation, str)
        self.assertTrue(representation.startswith("ParameterDictModel("))
        self.assertTrue(len(representation) > 0)
        # Check that parameter names appear in representation
        for name in self.parameters.keys():
            self.assertIn(name, representation)

    def test_getitem_existing_keys(self):
        """Test accessing existing parameters via __getitem__."""
        for name, param in self.parameters.items():
            retrieved = self.model[name]
            self.assertTrue(
                torch.allclose(retrieved, param),
                f"Parameter mismatch for {name}",
            )

    def test_getitem_nested_access(self):
        """Test accessing nested modules."""
        # Access intermediate nested module
        layer1 = self.model["layer1"]
        self.assertIsInstance(layer1, ParameterDictModel)

        # Access parameter through nested module
        weight = layer1.weight
        self.assertTrue(torch.allclose(weight, self.parameters["layer1.weight"]))

    def test_getitem_nonexistent_key(self):
        """Test accessing non-existent keys raises KeyError."""
        with self.assertRaises(KeyError):
            _ = self.model["nonexistent.key"]

        with self.assertRaises(KeyError):
            _ = self.model["layer3.weight"]

    def test_setitem_new_parameter(self):
        """Test adding new parameters via __setitem__."""
        new_param = nn.Parameter(torch.randn(5, 5))
        self.model["layer3.weight"] = new_param

        self.assertIn("layer3.weight", self.model)
        self.assertTrue(torch.allclose(self.model["layer3.weight"], new_param))

    def test_setitem_nested_parameter(self):
        """Test adding deeply nested parameters."""
        deep_param = nn.Parameter(torch.randn(3, 3))
        self.model["deep.nested.layer.weight"] = deep_param

        self.assertIn("deep.nested.layer.weight", self.model)
        self.assertTrue(
            torch.allclose(self.model["deep.nested.layer.weight"], deep_param)
        )

    def test_setitem_update_existing(self):
        """Test updating existing parameters."""
        old_value = self.model["layer1.weight"].clone()
        new_param = nn.Parameter(torch.randn(10, 10))
        self.model["layer1.weight"] = new_param

        self.assertTrue(torch.allclose(self.model["layer1.weight"], new_param))
        self.assertFalse(torch.allclose(self.model["layer1.weight"], old_value))

    def test_contains(self):
        """Test __contains__ method."""
        for name in self.parameters.keys():
            self.assertTrue(name in self.model, f"{name} should be in model")

        self.assertFalse("nonexistent" in self.model)
        self.assertFalse("layer3.weight" in self.model)

    def test_keys(self):
        """Test keys() method returns all parameter names."""
        keys = list(self.model.keys())
        self.assertEqual(len(keys), len(self.parameters))
        for name in self.parameters.keys():
            self.assertIn(name, keys)

    def test_items(self):
        """Test items() method returns (name, parameter) tuples."""
        items = list(self.model.items())
        self.assertEqual(len(items), len(self.parameters))

        for name, param in items:
            self.assertIn(name, self.parameters)
            self.assertTrue(
                torch.allclose(param, self.parameters[name]),
                f"Parameter mismatch for {name}",
            )

    def test_values(self):
        """Test values() method returns all parameters."""
        values = list(self.model.values())
        self.assertEqual(len(values), len(self.parameters))

        # Check that all original parameters are in values
        for original_param in self.parameters.values():
            found = any(
                v.shape == original_param.shape and torch.allclose(v, original_param)
                for v in values
            )
            self.assertTrue(found, "Original parameter not found in values")

    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.model), len(self.parameters))

        # Test with empty model
        empty_model = ParameterDictModel()
        self.assertEqual(len(empty_model), 0)

        # Test after adding parameter
        empty_model["new.param"] = nn.Parameter(torch.randn(5))
        self.assertEqual(len(empty_model), 1)

    def test_named_parameters(self):
        """Test that named_parameters() works correctly."""
        named_params = dict(self.model.named_parameters())

        self.assertEqual(len(named_params), len(self.parameters))
        for name, param in self.parameters.items():
            self.assertIn(name, named_params)
            self.assertTrue(torch.allclose(named_params[name], param))

    def test_state_dict(self):
        """Test that state_dict() includes all parameters."""
        state = self.model.state_dict()

        self.assertEqual(len(state), len(self.parameters))
        for name, param in self.parameters.items():
            self.assertIn(name, state)
            self.assertTrue(torch.allclose(state[name], param))

    def test_parameter_registration(self):
        """Test that parameters are properly registered with the module."""
        # Check that all parameters are registered
        for name, param in self.parameters.items():
            # Get the parameter using PyTorch's get_parameter
            retrieved = self.model.get_parameter(name)
            self.assertTrue(torch.allclose(retrieved, param))
            self.assertIsInstance(retrieved, nn.Parameter)

    def test_with_tensors_not_parameters(self):
        """Test that initialization with non-Parameter tensors raises assertion."""
        invalid_params = {
            "layer.weight": torch.randn(5, 5),  # Regular tensor, not Parameter
        }

        with self.assertRaises(AssertionError):
            ParameterDictModel(invalid_params)

    def test_nested_structure_creation(self):
        """Test that nested module structure is created correctly."""
        # Access nested module
        layer1 = self.model.layer1
        self.assertIsInstance(layer1, ParameterDictModel)

        # Check that nested module has the expected parameters
        self.assertTrue(hasattr(layer1, "weight"))
        self.assertTrue(hasattr(layer1, "bias"))

        layer2 = self.model.layer2
        self.assertIsInstance(layer2, ParameterDictModel)

    def test_iteration_consistency(self):
        """Test that keys, values, and items are consistent."""
        keys_list = list(self.model.keys())
        values_list = list(self.model.values())
        items_list = list(self.model.items())

        self.assertEqual(len(keys_list), len(values_list))
        self.assertEqual(len(keys_list), len(items_list))

        # Verify items match keys and values
        for i, (key, value) in enumerate(items_list):
            self.assertEqual(key, keys_list[i])
            self.assertTrue(torch.allclose(value, values_list[i]))


if __name__ == "__main__":
    unittest.main()
