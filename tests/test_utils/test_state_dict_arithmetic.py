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
        result = state_dict_binary_mask(
            state_dict_a, state_dict_c, "greater", strict=False
        )
        self.assertIn("weight", result)
        self.assertNotIn("bias", result)
        self.assertNotIn("different_key", result)
        expected_weight = torch.tensor([True, False, True])
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test with show_pbar=True (should work without errors)
        result = state_dict_binary_mask(
            state_dict_a, state_dict_b, "greater", show_pbar=True
        )
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

    def test_arithmetic_state_dict_creation(self):
        """Test ArithmeticStateDict creation and basic properties."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        # Test creation from dict
        regular_dict = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])}
        asd = ArithmeticStateDict(regular_dict)

        self.assertIsInstance(asd, ArithmeticStateDict)
        self.assertEqual(len(asd), 2)
        self.assertIn("weight", asd)
        self.assertIn("bias", asd)
        torch.testing.assert_close(asd["weight"], torch.tensor([1.0, 2.0]))

        # Test from_state_dict class method
        asd2 = ArithmeticStateDict.from_state_dict(regular_dict)
        self.assertIsInstance(asd2, ArithmeticStateDict)
        torch.testing.assert_close(asd2["weight"], asd["weight"])

    def test_arithmetic_state_dict_addition(self):
        """Test ArithmeticStateDict addition operations."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd1 = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5, 1.5])}
        )
        asd2 = ArithmeticStateDict(
            {"weight": torch.tensor([2.0, 1.0, 1.0]), "bias": torch.tensor([0.3, 0.7])}
        )

        # Test addition
        result = asd1 + asd2
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([3.0, 3.0, 4.0])
        expected_bias = torch.tensor([0.8, 2.2])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test right addition
        result2 = asd2 + asd1
        torch.testing.assert_close(result2["weight"], expected_weight)

        # Test in-place addition
        asd1_copy = asd1.clone()
        asd1_copy += asd2
        torch.testing.assert_close(asd1_copy["weight"], expected_weight)
        torch.testing.assert_close(asd1_copy["bias"], expected_bias)

        # Test type error
        with self.assertRaises(TypeError):
            asd1 + "invalid"

    def test_arithmetic_state_dict_subtraction(self):
        """Test ArithmeticStateDict subtraction operations."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd1 = ArithmeticStateDict(
            {"weight": torch.tensor([3.0, 4.0, 5.0]), "bias": torch.tensor([1.0, 2.0])}
        )
        asd2 = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 2.0]), "bias": torch.tensor([0.5, 0.5])}
        )

        # Test subtraction
        result = asd1 - asd2
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([2.0, 2.0, 3.0])
        expected_bias = torch.tensor([0.5, 1.5])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test right subtraction
        result2 = asd2 - asd1
        expected_weight2 = torch.tensor([-2.0, -2.0, -3.0])
        expected_bias2 = torch.tensor([-0.5, -1.5])
        torch.testing.assert_close(result2["weight"], expected_weight2)
        torch.testing.assert_close(result2["bias"], expected_bias2)

        # Test in-place subtraction
        asd1_copy = asd1.clone()
        asd1_copy -= asd2
        torch.testing.assert_close(asd1_copy["weight"], expected_weight)

    def test_arithmetic_state_dict_multiplication(self):
        """Test ArithmeticStateDict multiplication operations."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5, 1.5])}
        )
        asd2 = ArithmeticStateDict(
            {"weight": torch.tensor([2.0, 3.0, 2.0]), "bias": torch.tensor([4.0, 2.0])}
        )

        # Test scalar multiplication
        result = asd * 2.0
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([2.0, 4.0, 6.0])
        expected_bias = torch.tensor([1.0, 3.0])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test right scalar multiplication
        result2 = 3.0 * asd
        expected_weight2 = torch.tensor([3.0, 6.0, 9.0])
        torch.testing.assert_close(result2["weight"], expected_weight2)

        # Test Hadamard product (element-wise multiplication)
        result3 = asd * asd2
        expected_weight3 = torch.tensor([2.0, 6.0, 6.0])
        expected_bias3 = torch.tensor([2.0, 3.0])
        torch.testing.assert_close(result3["weight"], expected_weight3)
        torch.testing.assert_close(result3["bias"], expected_bias3)

        # Test in-place scalar multiplication
        asd_copy = asd.clone()
        asd_copy *= 2.0
        torch.testing.assert_close(asd_copy["weight"], expected_weight)

        # Test in-place Hadamard product
        asd_copy2 = asd.clone()
        asd_copy2 *= asd2
        torch.testing.assert_close(asd_copy2["weight"], expected_weight3)

    def test_arithmetic_state_dict_division(self):
        """Test ArithmeticStateDict division operations."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd = ArithmeticStateDict(
            {"weight": torch.tensor([4.0, 8.0, 12.0]), "bias": torch.tensor([2.0, 6.0])}
        )

        # Test scalar division
        result = asd / 2.0
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([2.0, 4.0, 6.0])
        expected_bias = torch.tensor([1.0, 3.0])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test in-place division
        asd_copy = asd.clone()
        asd_copy /= 2.0
        torch.testing.assert_close(asd_copy["weight"], expected_weight)

        # Test division by zero
        with self.assertRaises(ZeroDivisionError):
            asd / 0.0

        with self.assertRaises(ZeroDivisionError):
            asd_copy = asd.clone()
            asd_copy /= 0.0

        # Test type error
        with self.assertRaises(TypeError):
            asd / "invalid"

    def test_arithmetic_state_dict_power(self):
        """Test ArithmeticStateDict power operations."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([2.0, 4.0])}
        )

        # Test power operation
        result = asd**2
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([1.0, 4.0, 9.0])
        expected_bias = torch.tensor([4.0, 16.0])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test in-place power
        asd_copy = asd.clone()
        asd_copy **= 2
        torch.testing.assert_close(asd_copy["weight"], expected_weight)

        # Test type error
        with self.assertRaises(TypeError):
            asd ** "invalid"

    def test_arithmetic_state_dict_matmul(self):
        """Test ArithmeticStateDict matrix multiplication (Hadamard product)."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd1 = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5, 1.5])}
        )
        asd2 = ArithmeticStateDict(
            {"weight": torch.tensor([2.0, 3.0, 2.0]), "bias": torch.tensor([4.0, 2.0])}
        )

        # Test matmul (Hadamard product)
        result = asd1 @ asd2
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([2.0, 6.0, 6.0])
        expected_bias = torch.tensor([2.0, 3.0])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test right matmul
        result2 = asd2 @ asd1
        torch.testing.assert_close(result2["weight"], expected_weight)

        # Test type error
        with self.assertRaises(TypeError):
            asd1 @ "invalid"

    def test_arithmetic_state_dict_utility_methods(self):
        """Test ArithmeticStateDict utility methods."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd = ArithmeticStateDict(
            {
                "weight": torch.tensor([-1.0, 2.0, -3.0]),
                "bias": torch.tensor([4.0, 9.0]),
            }
        )

        # Test abs
        result = asd.abs()
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([1.0, 2.0, 3.0])
        expected_bias = torch.tensor([4.0, 9.0])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test sqrt
        asd_positive = ArithmeticStateDict(
            {
                "weight": torch.tensor([1.0, 4.0, 9.0]),
                "bias": torch.tensor([16.0, 25.0]),
            }
        )
        result = asd_positive.sqrt()
        expected_weight = torch.tensor([1.0, 2.0, 3.0])
        expected_bias = torch.tensor([4.0, 5.0])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test num_params
        num_params = asd.num_params()
        self.assertEqual(num_params, 5)  # 3 + 2 parameters

        # Test clone
        cloned = asd.clone()
        self.assertIsInstance(cloned, ArithmeticStateDict)
        self.assertIsNot(cloned, asd)
        torch.testing.assert_close(cloned["weight"], asd["weight"])

        # Modify original to ensure clone is independent
        asd["weight"][0] = 999.0
        self.assertNotEqual(cloned["weight"][0].item(), 999.0)

        # Test detach
        asd_with_grad = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0], requires_grad=True)}
        )
        detached = asd_with_grad.detach()
        self.assertFalse(detached["weight"].requires_grad)
        self.assertTrue(asd_with_grad["weight"].requires_grad)

    def test_arithmetic_state_dict_device_operations(self):
        """Test ArithmeticStateDict device operations."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5, 1.5])}
        )

        # Test to_device (CPU to CPU)
        result = asd.to_device("cpu")
        self.assertIsInstance(result, ArithmeticStateDict)
        self.assertEqual(result["weight"].device.type, "cpu")

        # Test in-place device transfer
        asd_copy = asd.clone()
        result_inplace = asd_copy.to_device("cpu", inplace=True)
        self.assertIs(result_inplace, asd_copy)

        # Skip CUDA tests if not available
        if torch.cuda.is_available():
            # Test CPU to CUDA
            cuda_result = asd.to_device("cuda")
            self.assertEqual(cuda_result["weight"].device.type, "cuda")

            # Test CUDA to CPU
            cpu_result = cuda_result.to_device("cpu")
            self.assertEqual(cpu_result["weight"].device.type, "cpu")

    def test_arithmetic_state_dict_class_methods(self):
        """Test ArithmeticStateDict class methods."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd1 = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5, 1.5])}
        )
        asd2 = ArithmeticStateDict(
            {"weight": torch.tensor([3.0, 2.0, 1.0]), "bias": torch.tensor([1.5, 0.5])}
        )

        # Test weighted_sum
        result = ArithmeticStateDict.weighted_sum([asd1, asd2], [0.3, 0.7])
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([2.4, 2.0, 1.6])  # 0.3*[1,2,3] + 0.7*[3,2,1]
        expected_bias = torch.tensor([1.2, 0.8])  # 0.3*[0.5,1.5] + 0.7*[1.5,0.5]
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test average
        result = ArithmeticStateDict.average([asd1, asd2])
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([2.0, 2.0, 2.0])  # ([1,2,3] + [3,2,1]) / 2
        expected_bias = torch.tensor([1.0, 1.0])  # ([0.5,1.5] + [1.5,0.5]) / 2
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

    def test_arithmetic_state_dict_error_handling(self):
        """Test ArithmeticStateDict error handling."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd1 = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])}
        )

        # Different keys should raise errors with strict operations
        asd2 = ArithmeticStateDict(
            {"weight": torch.tensor([3.0, 4.0]), "different_key": torch.tensor([1.0])}
        )

        # Test that operations with different keys raise ValueError
        with self.assertRaises(ValueError):
            asd1 + asd2

        with self.assertRaises(ValueError):
            asd1 - asd2

        with self.assertRaises(ValueError):
            asd1 @ asd2

        # Test type errors for invalid operations
        with self.assertRaises(TypeError):
            asd1 + "string"  # Can't add string to ArithmeticStateDict

        with self.assertRaises(TypeError):
            asd1 - "string"  # Can't subtract string

        with self.assertRaises(TypeError):
            "string" - asd1  # Can't subtract from string

        with self.assertRaises(TypeError):
            asd1 + [1, 2, 3]  # Can't add list

        with self.assertRaises(TypeError):
            None + asd1  # Can't add None

    def test_arithmetic_state_dict_sum_function(self):
        """Test ArithmeticStateDict with Python's built-in sum function."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        # Create test data
        models = [
            ArithmeticStateDict(
                {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])}
            ),
            ArithmeticStateDict(
                {"weight": torch.tensor([2.0, 3.0]), "bias": torch.tensor([1.0])}
            ),
            ArithmeticStateDict(
                {"weight": torch.tensor([3.0, 1.0]), "bias": torch.tensor([1.5])}
            ),
        ]

        # Test sum() function
        total = sum(models)
        self.assertIsInstance(total, ArithmeticStateDict)
        expected_weight = torch.tensor([6.0, 6.0])  # 1+2+3, 2+3+1
        expected_bias = torch.tensor([3.0])  # 0.5+1.0+1.5
        torch.testing.assert_close(total["weight"], expected_weight)
        torch.testing.assert_close(total["bias"], expected_bias)

        # Test ensemble averaging using sum
        ensemble = sum(models) / len(models)
        expected_avg_weight = torch.tensor([2.0, 2.0])  # 6/3, 6/3
        expected_avg_bias = torch.tensor([1.0])  # 3/3
        torch.testing.assert_close(ensemble["weight"], expected_avg_weight)
        torch.testing.assert_close(ensemble["bias"], expected_avg_bias)

        # Test sum with empty list (should work with sum([], start))
        empty_sum = sum(
            [],
            ArithmeticStateDict(
                {"weight": torch.tensor([0.0]), "bias": torch.tensor([0.0])}
            ),
        )
        self.assertIsInstance(empty_sum, ArithmeticStateDict)
        torch.testing.assert_close(empty_sum["weight"], torch.tensor([0.0]))

    def test_arithmetic_state_dict_scalar_operations(self):
        """Test ArithmeticStateDict operations with scalars."""
        from fusion_bench.utils.state_dict_arithmetic import ArithmeticStateDict

        asd = ArithmeticStateDict(
            {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5, 1.5])}
        )

        # Test scalar addition
        result = asd + 2.0
        self.assertIsInstance(result, ArithmeticStateDict)
        expected_weight = torch.tensor([3.0, 4.0, 5.0])
        expected_bias = torch.tensor([2.5, 3.5])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test right scalar addition
        result = 3.0 + asd
        expected_weight = torch.tensor([4.0, 5.0, 6.0])
        expected_bias = torch.tensor([3.5, 4.5])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test scalar subtraction
        result = asd - 1.0
        expected_weight = torch.tensor([0.0, 1.0, 2.0])
        expected_bias = torch.tensor([-0.5, 0.5])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test right scalar subtraction
        result = 5.0 - asd
        expected_weight = torch.tensor([4.0, 3.0, 2.0])
        expected_bias = torch.tensor([4.5, 3.5])
        torch.testing.assert_close(result["weight"], expected_weight)
        torch.testing.assert_close(result["bias"], expected_bias)

        # Test in-place scalar addition
        asd_copy = asd.clone()
        asd_copy += 2.5
        expected_weight = torch.tensor([3.5, 4.5, 5.5])
        expected_bias = torch.tensor([3.0, 4.0])
        torch.testing.assert_close(asd_copy["weight"], expected_weight)
        torch.testing.assert_close(asd_copy["bias"], expected_bias)

        # Test in-place scalar subtraction
        asd_copy = asd.clone()
        asd_copy -= 0.5
        expected_weight = torch.tensor([0.5, 1.5, 2.5])
        expected_bias = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(asd_copy["weight"], expected_weight)
        torch.testing.assert_close(asd_copy["bias"], expected_bias)

        # Test with integers
        result = asd + 2
        expected_weight = torch.tensor([3.0, 4.0, 5.0])
        torch.testing.assert_close(result["weight"], expected_weight)

        # Test with negative scalars
        result = asd + (-1.5)
        expected_weight = torch.tensor([-0.5, 0.5, 1.5])
        torch.testing.assert_close(result["weight"], expected_weight)


if __name__ == "__main__":
    unittest.main()
