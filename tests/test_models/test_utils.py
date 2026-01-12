import unittest

import torch
import torch.nn as nn

from fusion_bench.models.utils import (
    get_target_state_dict,
    load_state_dict_into_target_modules,
)


class SimpleModel(nn.Module):
    """A simple model with multiple submodules for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 40)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class TestModelStateUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        # Initialize with random weights
        torch.manual_seed(42)
        for param in self.model.parameters():
            param.data.normal_(0, 1)

    def test_get_target_state_dict_full_model(self):
        """Test getting state dict of the entire model."""
        state_dict = get_target_state_dict(self.model)
        expected_keys = set(self.model.state_dict().keys())
        self.assertEqual(set(state_dict.keys()), expected_keys)

    def test_get_target_state_dict_single_module(self):
        """Test getting state dict of a single target module."""
        state_dict = get_target_state_dict(self.model, target_modules="layer1")
        expected_keys = {"layer1.weight", "layer1.bias"}
        self.assertEqual(set(state_dict.keys()), expected_keys)

    def test_get_target_state_dict_multiple_modules(self):
        """Test getting state dict of multiple target modules."""
        state_dict = get_target_state_dict(
            self.model, target_modules=["layer1", "layer3"]
        )
        expected_keys = {
            "layer1.weight",
            "layer1.bias",
            "layer3.weight",
            "layer3.bias",
        }
        self.assertEqual(set(state_dict.keys()), expected_keys)

    def test_get_target_state_dict_with_prefix(self):
        """Test getting state dict with a custom prefix."""
        state_dict = get_target_state_dict(
            self.model, target_modules="layer1", prefix="model."
        )
        expected_keys = {"model.layer1.weight", "model.layer1.bias"}
        self.assertEqual(set(state_dict.keys()), expected_keys)

    def test_get_target_state_dict_fusion_bench_attribute(self):
        """Test getting state dict using _fusion_bench_target_modules attribute."""
        self.model._fusion_bench_target_modules = ["layer1", "layer2"]
        state_dict = get_target_state_dict(self.model)
        expected_keys = {
            "layer1.weight",
            "layer1.bias",
            "layer2.weight",
            "layer2.bias",
        }
        self.assertEqual(set(state_dict.keys()), expected_keys)

    def test_load_state_dict_into_target_modules_full_model(self):
        """Test loading state dict into the entire model."""
        original_state_dict = self.model.state_dict()
        new_model = SimpleModel()

        # Load the state dict
        result = load_state_dict_into_target_modules(new_model, original_state_dict)

        # Check that loading was successful
        self.assertEqual(len(result.missing_keys), 0)
        self.assertEqual(len(result.unexpected_keys), 0)

        # Verify parameters match
        for key in original_state_dict:
            torch.testing.assert_close(
                new_model.state_dict()[key], original_state_dict[key]
            )

    def test_load_state_dict_into_target_modules_single_module(self):
        """Test loading state dict into a single target module."""
        # Get state dict of layer1
        layer1_state_dict = get_target_state_dict(self.model, target_modules="layer1")

        # Create a new model and load only layer1
        new_model = SimpleModel()
        original_layer2_weight = new_model.layer2.weight.clone()

        result = load_state_dict_into_target_modules(
            new_model, layer1_state_dict, target_modules="layer1"
        )

        # Check that loading was successful
        self.assertEqual(len(result.missing_keys), 0)
        self.assertEqual(len(result.unexpected_keys), 0)

        # Verify layer1 parameters match
        torch.testing.assert_close(new_model.layer1.weight, self.model.layer1.weight)
        torch.testing.assert_close(new_model.layer1.bias, self.model.layer1.bias)

        # Verify layer2 parameters are unchanged
        torch.testing.assert_close(new_model.layer2.weight, original_layer2_weight)

    def test_load_state_dict_into_target_modules_multiple_modules(self):
        """Test loading state dict into multiple target modules."""
        # Get state dict of layer1 and layer3
        layer1_state_dict = get_target_state_dict(self.model, target_modules="layer1")
        layer3_state_dict = get_target_state_dict(self.model, target_modules="layer3")

        # Create a new model and load layer1 and layer3
        new_model = SimpleModel()
        original_layer2_weight = new_model.layer2.weight.clone()

        # Load each module individually
        result1 = load_state_dict_into_target_modules(
            new_model, layer1_state_dict, target_modules="layer1"
        )
        result3 = load_state_dict_into_target_modules(
            new_model, layer3_state_dict, target_modules="layer3"
        )

        # Check that loading was successful
        self.assertEqual(len(result1.missing_keys), 0)
        self.assertEqual(len(result1.unexpected_keys), 0)
        self.assertEqual(len(result3.missing_keys), 0)
        self.assertEqual(len(result3.unexpected_keys), 0)

        # Verify layer1 and layer3 parameters match
        torch.testing.assert_close(new_model.layer1.weight, self.model.layer1.weight)
        torch.testing.assert_close(new_model.layer1.bias, self.model.layer1.bias)
        torch.testing.assert_close(new_model.layer3.weight, self.model.layer3.weight)
        torch.testing.assert_close(new_model.layer3.bias, self.model.layer3.bias)

        # Verify layer2 parameters are unchanged
        torch.testing.assert_close(new_model.layer2.weight, original_layer2_weight)

    def test_load_state_dict_into_target_modules_fusion_bench_attribute(self):
        """Test loading state dict using _fusion_bench_target_modules attribute."""
        # Get state dict of layer1 and layer2 separately
        layer1_state_dict = get_target_state_dict(self.model, target_modules="layer1")
        layer2_state_dict = get_target_state_dict(self.model, target_modules="layer2")

        # Create a new model
        new_model = SimpleModel()

        # Load each module individually
        result1 = load_state_dict_into_target_modules(
            new_model, layer1_state_dict, target_modules="layer1"
        )
        result2 = load_state_dict_into_target_modules(
            new_model, layer2_state_dict, target_modules="layer2"
        )

        # Check that loading was successful
        self.assertEqual(len(result1.missing_keys), 0)
        self.assertEqual(len(result1.unexpected_keys), 0)
        self.assertEqual(len(result2.missing_keys), 0)
        self.assertEqual(len(result2.unexpected_keys), 0)

        # Verify parameters match
        torch.testing.assert_close(new_model.layer1.weight, self.model.layer1.weight)
        torch.testing.assert_close(new_model.layer1.bias, self.model.layer1.bias)
        torch.testing.assert_close(new_model.layer2.weight, self.model.layer2.weight)
        torch.testing.assert_close(new_model.layer2.bias, self.model.layer2.bias)

    def test_load_state_dict_strict_mode(self):
        """Test strict mode when loading state dict."""
        # Create a state dict with an extra key
        state_dict = get_target_state_dict(self.model, target_modules="layer1")
        state_dict["layer1.extra_param"] = torch.randn(10)

        new_model = SimpleModel()

        # Should raise an error in strict mode
        with self.assertRaises(RuntimeError):
            load_state_dict_into_target_modules(
                new_model, state_dict, target_modules="layer1", strict=True
            )

    def test_load_state_dict_non_strict_mode(self):
        """Test non-strict mode when loading state dict."""
        # Create a state dict with an extra key
        state_dict = get_target_state_dict(self.model, target_modules="layer1")
        state_dict["layer1.extra_param"] = torch.randn(10)

        new_model = SimpleModel()

        # Should not raise an error in non-strict mode
        result = load_state_dict_into_target_modules(
            new_model, state_dict, target_modules="layer1", strict=False
        )

        # Should report the unexpected key (note: the prefix is stripped)
        self.assertIn("extra_param", result.unexpected_keys)

    def test_roundtrip_state_dict(self):
        """Test that get and load operations are inverses of each other."""
        # Get state dict of specific modules separately
        layer1_state_dict = get_target_state_dict(self.model, target_modules="layer1")
        layer2_state_dict = get_target_state_dict(self.model, target_modules="layer2")

        # Create a new model and load the state dict
        new_model = SimpleModel()
        load_state_dict_into_target_modules(
            new_model, layer1_state_dict, target_modules="layer1"
        )
        load_state_dict_into_target_modules(
            new_model, layer2_state_dict, target_modules="layer2"
        )

        # Get state dict again and compare for each module
        new_layer1_state_dict = get_target_state_dict(
            new_model, target_modules="layer1"
        )
        new_layer2_state_dict = get_target_state_dict(
            new_model, target_modules="layer2"
        )

        # All keys should match
        self.assertEqual(
            set(layer1_state_dict.keys()), set(new_layer1_state_dict.keys())
        )
        self.assertEqual(
            set(layer2_state_dict.keys()), set(new_layer2_state_dict.keys())
        )

        # All values should match
        for key in layer1_state_dict:
            torch.testing.assert_close(
                layer1_state_dict[key], new_layer1_state_dict[key]
            )
        for key in layer2_state_dict:
            torch.testing.assert_close(
                layer2_state_dict[key], new_layer2_state_dict[key]
            )


if __name__ == "__main__":
    unittest.main()
