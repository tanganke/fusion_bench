import timeit
import unittest
import warnings

import torch
from torch import nn

from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import *


def create_test_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
    )


def create_simple_state_dict():
    """Create a simple state dict for testing."""
    return {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.1, 0.2])}


def create_incompatible_state_dict():
    """Create a state dict with different shapes for testing validation."""
    return {
        "weight": torch.tensor([1.0, 2.0]),  # Different shape
        "bias": torch.tensor([0.1, 0.2]),
    }


class TestStateDictArithmetic(unittest.TestCase):
    def setUp(self):
        self.model_1 = create_test_model()
        self.model_2 = create_test_model()
        self.model_3 = create_test_model()
        if torch.cuda.is_available():
            self.model_1.cuda()
            self.model_2.cuda()
            self.model_3.cuda()

        # Simple test data
        self.simple_state_dict_1 = create_simple_state_dict()
        self.simple_state_dict_2 = create_simple_state_dict()
        self.incompatible_state_dict = create_incompatible_state_dict()
        self.missing_key_state_dict = {"weight": torch.tensor([1.0, 2.0, 3.0])}

    def test_validation_functions(self):
        """Test validation through public API functions."""
        # Test empty list validation through state_dict_sum
        with self.assertRaises(ValueError):
            state_dict_sum([])

        # Test state_dicts_check_keys function
        state_dicts_check_keys([self.simple_state_dict_1, self.simple_state_dict_2])

        # Test incompatible state dicts through state_dict_sum
        with self.assertRaises(ValueError):
            state_dict_sum([self.simple_state_dict_1, self.incompatible_state_dict])

        # Test missing keys through state_dict_sum
        with self.assertRaises(ValueError):
            state_dict_sum([self.simple_state_dict_1, self.missing_key_state_dict])

    def test_state_dict_sum_validation(self):
        """Test state_dict_sum with validation."""
        # Valid case
        result = state_dict_sum([self.simple_state_dict_1, self.simple_state_dict_2])
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Empty list should raise error
        with self.assertRaises(ValueError):
            state_dict_sum([])

        # Incompatible shapes should raise error
        with self.assertRaises(ValueError):
            state_dict_sum([self.simple_state_dict_1, self.incompatible_state_dict])

    def test_state_dict_avg_validation(self):
        """Test state_dict_avg with validation."""
        # Valid case
        result = state_dict_avg([self.simple_state_dict_1, self.simple_state_dict_2])
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Check that it's actually averaging
        expected_weight = (
            self.simple_state_dict_1["weight"] + self.simple_state_dict_2["weight"]
        ) / 2
        torch.testing.assert_close(result["weight"], expected_weight)

        # Empty list should raise error
        with self.assertRaises(ValueError):
            state_dict_avg([])

    def test_state_dict_sub_validation(self):
        """Test state_dict_sub with enhanced validation."""
        # Valid case with strict=True
        result = state_dict_sub(
            self.simple_state_dict_1, self.simple_state_dict_2, strict=True
        )
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Missing key with strict=True should raise error
        with self.assertRaises(ValueError):
            state_dict_sub(
                self.simple_state_dict_1, self.missing_key_state_dict, strict=True
            )

        # Shape mismatch with strict=True should raise error
        with self.assertRaises(ValueError):
            state_dict_sub(
                self.simple_state_dict_1, self.incompatible_state_dict, strict=True
            )

        # Non-strict mode should work with common keys only
        result = state_dict_sub(
            self.simple_state_dict_1, self.missing_key_state_dict, strict=False
        )
        self.assertIn("weight", result)
        self.assertNotIn("bias", result)

    def test_state_dict_add_validation(self):
        """Test state_dict_add with validation."""
        # Valid case with strict=True
        result = state_dict_add(
            self.simple_state_dict_1, self.simple_state_dict_2, strict=True
        )
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Non-strict mode should work with common keys
        result = state_dict_add(
            self.simple_state_dict_1, self.missing_key_state_dict, strict=False
        )
        self.assertIn("weight", result)
        self.assertNotIn("bias", result)

    def test_state_dict_hadamard_product_validation(self):
        """Test state_dict_hadamard_product with validation."""
        # Valid case
        result = state_dict_hadamard_product(
            self.simple_state_dict_1, self.simple_state_dict_2
        )
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Check that it's actually computing element-wise product
        expected_weight = (
            self.simple_state_dict_1["weight"] * self.simple_state_dict_2["weight"]
        )
        torch.testing.assert_close(result["weight"], expected_weight)

        # Missing key should raise error
        with self.assertRaises(ValueError):
            state_dict_hadamard_product(
                self.simple_state_dict_1, self.missing_key_state_dict
            )

        # Shape mismatch should raise error
        with self.assertRaises(ValueError):
            state_dict_hadamard_product(
                self.simple_state_dict_1, self.incompatible_state_dict
            )

    def test_state_dict_binary_mask(self):
        """Test state_dict_binary_mask function."""
        state_dict_a = {
            "weight": torch.tensor([3.0, 1.0, 5.0]),
            "bias": torch.tensor([0.5, 0.1]),
        }
        state_dict_b = {
            "weight": torch.tensor([1.0, 2.0, 3.0]),
            "bias": torch.tensor([0.3, 0.2]),
        }

        # Test greater comparison
        result = state_dict_binary_mask(state_dict_a, state_dict_b, "greater")
        expected_weight = torch.tensor([True, False, True])
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test with custom function
        result = state_dict_binary_mask(state_dict_a, state_dict_b, lambda x, y: x < y)
        expected_weight = torch.tensor([False, True, False])
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test with strict=True (should raise error for different keys)
        state_dict_c = {
            "weight": torch.tensor([1.0, 2.0, 3.0]),
            "different_key": torch.tensor([0.3, 0.2]),
        }
        with self.assertRaises(ValueError):
            state_dict_binary_mask(state_dict_a, state_dict_c, "greater", strict=True)

        # Test with strict=False (should work with common keys only)
        result = state_dict_binary_mask(state_dict_a, state_dict_c, "greater", strict=False)
        self.assertIn("weight", result)
        self.assertNotIn("bias", result)
        self.assertNotIn("different_key", result)
        expected_weight = torch.tensor([True, False, True])
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test with show_pbar=True (should work without errors)
        result = state_dict_binary_mask(state_dict_a, state_dict_b, "greater", show_pbar=True)
        expected_weight = torch.tensor([True, False, True])
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test invalid compare_fn
        with self.assertRaises(ValueError):
            state_dict_binary_mask(state_dict_a, state_dict_b, "invalid")

    def test_scalar_operations(self):
        """Test scalar operations."""
        scalar = 2.0

        # Test state_dict_mul
        result = state_dict_mul(self.simple_state_dict_1, scalar)
        expected_weight = self.simple_state_dict_1["weight"] * scalar
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test state_dict_div
        result = state_dict_div(self.simple_state_dict_1, scalar)
        expected_weight = self.simple_state_dict_1["weight"] / scalar
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test division by zero
        with self.assertRaises(ZeroDivisionError):
            state_dict_div(self.simple_state_dict_1, 0.0)

        # Test state_dict_add_scalar
        result = state_dict_add_scalar(self.simple_state_dict_1, scalar)
        expected_weight = self.simple_state_dict_1["weight"] + scalar
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test state_dict_power
        result = state_dict_power(self.simple_state_dict_1, 2.0)
        expected_weight = self.simple_state_dict_1["weight"] ** 2.0
        torch.testing.assert_close(result["weight"], expected_weight)

    def test_utility_functions(self):
        """Test utility functions."""
        # Test num_params_of_state_dict
        num_params = num_params_of_state_dict(self.simple_state_dict_1)
        expected = 3 + 2  # weight has 3 elements, bias has 2
        self.assertEqual(num_params, expected)

        # Test state_dict_flatten
        flattened = state_dict_flatten(self.simple_state_dict_1)
        expected_length = 3 + 2  # Total elements
        self.assertEqual(len(flattened), expected_length)

        # Test to_device
        result = to_device(self.simple_state_dict_1, "cpu")
        self.assertEqual(result["weight"].device.type, "cpu")

    # Performance tests with real models
    def test_state_dict_sum(self):
        """Performance test for state_dict_sum with real models."""
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        state_dict_3 = self.model_3.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_sum([state_dict_1, state_dict_2, state_dict_3]),
            number=10,
        )
        print(f"Time taken for state_dict_sum: {time_taken} seconds")

    def test_state_dict_avg(self):
        """Performance test for state_dict_avg with real models."""
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        state_dict_3 = self.model_3.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_avg([state_dict_1, state_dict_2, state_dict_3]),
            number=10,
        )
        print(f"Time taken for state_dict_avg: {time_taken} seconds")

    def test_state_dict_sub(self):
        """Performance test for state_dict_sub with real models."""
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_sub(state_dict_1, state_dict_2),
            number=10,
        )
        print(f"Time taken for state_dict_sub: {time_taken} seconds")

    def test_state_dict_add(self):
        """Performance test for state_dict_add with real models."""
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_add(state_dict_1, state_dict_2),
            number=10,
        )
        print(f"Time taken for state_dict_add: {time_taken} seconds")

    def test_state_dict_mul(self):
        """Performance test for state_dict_mul with real models."""
        state_dict_1 = self.model_1.state_dict()
        scalar = 2.0
        time_taken = timeit.timeit(
            lambda: state_dict_mul(state_dict_1, scalar),
            number=10,
        )
        print(f"Time taken for state_dict_mul: {time_taken} seconds")

    def test_interpolation_and_weighted_sum(self):
        """Test interpolation and weighted sum functions."""
        state_dicts = [self.simple_state_dict_1, self.simple_state_dict_2]

        # Test interpolation
        scalars = [0.3, 0.7]
        result = state_dict_interpolation(state_dicts, scalars)
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Test weighted sum
        weights = [2.0, 3.0]
        result = state_dict_weighted_sum(state_dicts, weights)
        self.assertIn("weight", result)
        self.assertIn("bias", result)

        # Test mismatched lengths
        with self.assertRaises(ValueError):
            state_dict_interpolation(state_dicts, [0.5])  # Wrong number of scalars

        with self.assertRaises(ValueError):
            state_dict_weighted_sum(state_dicts, [1.0])  # Wrong number of weights


if __name__ == "__main__":
    unittest.main()
