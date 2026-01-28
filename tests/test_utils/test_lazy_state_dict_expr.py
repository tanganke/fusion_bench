"""
Unit tests for LazyStateDictExpr and its subclasses.

This module tests the lazy evaluation functionality for state dict operations,
including StateDictLeaf, UnaryOp, BinaryOp, and the ensure_expr helper.
"""

import unittest
from collections import OrderedDict

import torch

from fusion_bench.utils.state_dict_arithmetic import (
    BinaryOp,
    LazyStateDictExpr,
    StateDictLeaf,
    UnaryOp,
    ensure_expr,
)


def create_simple_state_dict():
    """Create a simple state dict for testing."""
    return OrderedDict(
        [
            ("weight", torch.tensor([1.0, 2.0, 3.0])),
            ("bias", torch.tensor([0.1, 0.2])),
        ]
    )


def create_another_state_dict():
    """Create another state dict with same structure for testing."""
    return OrderedDict(
        [
            ("weight", torch.tensor([2.0, 3.0, 4.0])),
            ("bias", torch.tensor([0.3, 0.4])),
        ]
    )


class TestStateDictLeaf(unittest.TestCase):
    """Test cases for StateDictLeaf class."""

    def setUp(self):
        self.state_dict = create_simple_state_dict()
        self.leaf = StateDictLeaf(self.state_dict)

    def test_init(self):
        """Test StateDictLeaf initialization."""
        self.assertIsInstance(self.leaf, LazyStateDictExpr)
        self.assertIsInstance(self.leaf, StateDictLeaf)

    def test_getitem(self):
        """Test accessing items via __getitem__."""
        self.assertTrue(torch.equal(self.leaf["weight"], self.state_dict["weight"]))
        self.assertTrue(torch.equal(self.leaf["bias"], self.state_dict["bias"]))

    def test_getitem_missing_key(self):
        """Test accessing non-existent key raises KeyError."""
        with self.assertRaises(KeyError):
            _ = self.leaf["nonexistent"]

    def test_iter(self):
        """Test iteration over keys."""
        keys = list(self.leaf)
        self.assertEqual(keys, list(self.state_dict.keys()))

    def test_len(self):
        """Test length of state dict."""
        self.assertEqual(len(self.leaf), len(self.state_dict))

    def test_mapping_protocol(self):
        """Test that StateDictLeaf implements Mapping protocol."""
        # Test keys()
        self.assertEqual(set(self.leaf.keys()), set(self.state_dict.keys()))

        # Test values() - check all values are tensors
        for value in self.leaf.values():
            self.assertIsInstance(value, torch.Tensor)

        # Test items()
        for key, value in self.leaf.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, torch.Tensor)
            self.assertTrue(torch.equal(value, self.state_dict[key]))

    def test_empty_state_dict(self):
        """Test StateDictLeaf with empty state dict."""
        empty_leaf = StateDictLeaf({})
        self.assertEqual(len(empty_leaf), 0)
        self.assertEqual(list(empty_leaf), [])


class TestUnaryOp(unittest.TestCase):
    """Test cases for UnaryOp class."""

    def setUp(self):
        self.state_dict = create_simple_state_dict()
        self.leaf = StateDictLeaf(self.state_dict)

    def test_unary_op_basic(self):
        """Test basic unary operation."""
        # Double all values
        double_op = UnaryOp(lambda x: x * 2, self.leaf)

        self.assertTrue(torch.equal(double_op["weight"], self.state_dict["weight"] * 2))
        self.assertTrue(torch.equal(double_op["bias"], self.state_dict["bias"] * 2))

    def test_unary_op_keys_and_length(self):
        """Test that unary op preserves keys and length."""
        square_op = UnaryOp(lambda x: x**2, self.leaf)

        self.assertEqual(len(square_op), len(self.leaf))
        self.assertEqual(list(square_op.keys()), list(self.leaf.keys()))

    def test_unary_op_composition(self):
        """Test composing multiple unary operations."""
        # (x * 2) + 1
        double_op = UnaryOp(lambda x: x * 2, self.leaf)
        add_one_op = UnaryOp(lambda x: x + 1, double_op)

        expected_weight = self.state_dict["weight"] * 2 + 1
        self.assertTrue(torch.equal(add_one_op["weight"], expected_weight))

    def test_unary_op_lazy_evaluation(self):
        """Test that unary operations are evaluated lazily."""
        call_count = 0

        def counting_op(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        lazy_op = UnaryOp(counting_op, self.leaf)

        # Creating the operation shouldn't call the function
        self.assertEqual(call_count, 0)

        # Accessing a key should call the function once
        _ = lazy_op["weight"]
        self.assertEqual(call_count, 1)

        # Accessing the same key again should call again (not cached)
        _ = lazy_op["weight"]
        self.assertEqual(call_count, 2)

    def test_unary_op_torch_functions(self):
        """Test unary operations with torch functions."""
        # Test abs
        state_dict_negative = OrderedDict(
            [
                ("weight", torch.tensor([-1.0, -2.0, -3.0])),
                ("bias", torch.tensor([-0.1, -0.2])),
            ]
        )
        leaf_negative = StateDictLeaf(state_dict_negative)
        abs_op = UnaryOp(torch.abs, leaf_negative)

        self.assertTrue(
            torch.equal(abs_op["weight"], torch.abs(state_dict_negative["weight"]))
        )

        # Test sqrt
        sqrt_op = UnaryOp(torch.sqrt, self.leaf)
        self.assertTrue(
            torch.allclose(sqrt_op["weight"], torch.sqrt(self.state_dict["weight"]))
        )


class TestBinaryOp(unittest.TestCase):
    """Test cases for BinaryOp class."""

    def setUp(self):
        self.state_dict_1 = create_simple_state_dict()
        self.state_dict_2 = create_another_state_dict()
        self.leaf_1 = StateDictLeaf(self.state_dict_1)
        self.leaf_2 = StateDictLeaf(self.state_dict_2)

    def test_binary_op_add(self):
        """Test binary addition operation."""
        add_op = BinaryOp(torch.add, self.leaf_1, self.leaf_2)

        expected_weight = self.state_dict_1["weight"] + self.state_dict_2["weight"]
        self.assertTrue(torch.equal(add_op["weight"], expected_weight))

        expected_bias = self.state_dict_1["bias"] + self.state_dict_2["bias"]
        self.assertTrue(torch.equal(add_op["bias"], expected_bias))

    def test_binary_op_sub(self):
        """Test binary subtraction operation."""
        sub_op = BinaryOp(torch.sub, self.leaf_1, self.leaf_2)

        expected_weight = self.state_dict_1["weight"] - self.state_dict_2["weight"]
        self.assertTrue(torch.equal(sub_op["weight"], expected_weight))

    def test_binary_op_mul(self):
        """Test binary multiplication operation."""
        mul_op = BinaryOp(torch.mul, self.leaf_1, self.leaf_2)

        expected_weight = self.state_dict_1["weight"] * self.state_dict_2["weight"]
        self.assertTrue(torch.equal(mul_op["weight"], expected_weight))

    def test_binary_op_keys_and_length(self):
        """Test that binary op uses left operand's keys."""
        mul_op = BinaryOp(torch.mul, self.leaf_1, self.leaf_2)

        # Length and keys should match left operand
        self.assertEqual(len(mul_op), len(self.leaf_1))
        self.assertEqual(list(mul_op.keys()), list(self.leaf_1.keys()))

    def test_binary_op_composition(self):
        """Test composing multiple binary operations."""
        # (a + b) * c
        state_dict_3 = create_simple_state_dict()
        leaf_3 = StateDictLeaf(state_dict_3)

        add_op = BinaryOp(torch.add, self.leaf_1, self.leaf_2)
        mul_op = BinaryOp(torch.mul, add_op, leaf_3)

        expected = (
            self.state_dict_1["weight"] + self.state_dict_2["weight"]
        ) * state_dict_3["weight"]
        self.assertTrue(torch.equal(mul_op["weight"], expected))

    def test_binary_op_lazy_evaluation(self):
        """Test that binary operations are evaluated lazily."""
        call_count = 0

        def counting_add(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        lazy_op = BinaryOp(counting_add, self.leaf_1, self.leaf_2)

        # Creating the operation shouldn't call the function
        self.assertEqual(call_count, 0)

        # Accessing a key should call the function once
        _ = lazy_op["weight"]
        self.assertEqual(call_count, 1)

    def test_binary_op_mismatched_keys(self):
        """Test binary op with mismatched keys."""
        # Create state dict with missing key
        incomplete_state_dict = OrderedDict([("weight", torch.tensor([1.0, 2.0, 3.0]))])
        incomplete_leaf = StateDictLeaf(incomplete_state_dict)

        add_op = BinaryOp(torch.add, self.leaf_1, incomplete_leaf)

        # Should work for matching keys
        _ = add_op["weight"]

        # Should raise KeyError for non-matching keys
        with self.assertRaises(KeyError):
            _ = add_op["bias"]


class TestLazyStateDictExprArithmetic(unittest.TestCase):
    """Test arithmetic operations on LazyStateDictExpr."""

    def setUp(self):
        self.state_dict_1 = create_simple_state_dict()
        self.state_dict_2 = create_another_state_dict()
        self.leaf_1 = StateDictLeaf(self.state_dict_1)
        self.leaf_2 = StateDictLeaf(self.state_dict_2)

    def test_add_operator(self):
        """Test __add__ operator."""
        result = self.leaf_1 + self.leaf_2

        self.assertIsInstance(result, BinaryOp)
        expected = self.state_dict_1["weight"] + self.state_dict_2["weight"]
        self.assertTrue(torch.equal(result["weight"], expected))

    def test_sub_operator(self):
        """Test __sub__ operator."""
        result = self.leaf_1 - self.leaf_2

        self.assertIsInstance(result, BinaryOp)
        expected = self.state_dict_1["weight"] - self.state_dict_2["weight"]
        self.assertTrue(torch.equal(result["weight"], expected))

    def test_mul_operator(self):
        """Test __mul__ operator with scalar."""
        result = self.leaf_1 * 2.0

        self.assertIsInstance(result, UnaryOp)
        expected = self.state_dict_1["weight"] * 2.0
        self.assertTrue(torch.equal(result["weight"], expected))

    def test_rmul_operator(self):
        """Test __rmul__ operator with scalar."""
        result = 3.0 * self.leaf_1

        self.assertIsInstance(result, UnaryOp)
        expected = self.state_dict_1["weight"] * 3.0
        self.assertTrue(torch.equal(result["weight"], expected))

    def test_truediv_operator(self):
        """Test __truediv__ operator."""
        result = self.leaf_1 / 2.0

        self.assertIsInstance(result, UnaryOp)
        expected = self.state_dict_1["weight"] / 2.0
        self.assertTrue(torch.equal(result["weight"], expected))

    def test_complex_expression(self):
        """Test complex arithmetic expression."""
        # (a + b) * 2 - a / 3
        result = (self.leaf_1 + self.leaf_2) * 2.0 - self.leaf_1 / 3.0

        expected = (
            self.state_dict_1["weight"] + self.state_dict_2["weight"]
        ) * 2.0 - self.state_dict_1["weight"] / 3.0
        self.assertTrue(torch.allclose(result["weight"], expected))

    def test_chained_operations(self):
        """Test chaining multiple operations."""
        # ((a + b) - a) * 2
        result = ((self.leaf_1 + self.leaf_2) - self.leaf_1) * 2.0

        expected = (
            (self.state_dict_1["weight"] + self.state_dict_2["weight"])
            - self.state_dict_1["weight"]
        ) * 2.0
        self.assertTrue(torch.equal(result["weight"], expected))


class TestMaterialize(unittest.TestCase):
    """Test materialize functionality."""

    def setUp(self):
        self.state_dict = create_simple_state_dict()
        self.leaf = StateDictLeaf(self.state_dict)

    def test_materialize_basic(self):
        """Test basic materialize."""
        result = (self.leaf * 2.0).materialize()

        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(self.state_dict.keys()))

        expected_weight = self.state_dict["weight"] * 2.0
        self.assertTrue(torch.equal(result["weight"], expected_weight))

    def test_materialize_complex_expression(self):
        """Test materialize with complex expression."""
        state_dict_2 = create_another_state_dict()
        leaf_2 = StateDictLeaf(state_dict_2)

        expr = (self.leaf + leaf_2) * 2.0 - self.leaf / 3.0
        result = expr.materialize()

        self.assertIsInstance(result, dict)
        for key in self.state_dict.keys():
            self.assertIn(key, result)
            self.assertIsInstance(result[key], torch.Tensor)

    def test_materialize_with_device_cpu(self):
        """Test materialize with device parameter (CPU)."""
        result = self.leaf.materialize(device="cpu")

        for key, tensor in result.items():
            self.assertEqual(tensor.device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_materialize_with_device_cuda(self):
        """Test materialize with device parameter (CUDA)."""
        result = self.leaf.materialize(device="cuda")

        for key, tensor in result.items():
            self.assertEqual(tensor.device.type, "cuda")

    def test_materialize_with_dtype(self):
        """Test materialize with dtype parameter."""
        result = self.leaf.materialize(dtype=torch.float64)

        for key, tensor in result.items():
            self.assertEqual(tensor.dtype, torch.float64)

    def test_materialize_with_copy(self):
        """Test materialize with copy parameter."""
        result = self.leaf.materialize(copy=True)

        # Verify it's a copy (modifying result shouldn't affect original)
        original_value = self.state_dict["weight"].clone()
        result["weight"].fill_(999.0)
        self.assertTrue(torch.equal(self.state_dict["weight"], original_value))

    def test_materialize_preserves_order(self):
        """Test that materialize preserves key order."""
        result = self.leaf.materialize()

        self.assertEqual(list(result.keys()), list(self.state_dict.keys()))


class TestEnsureExpr(unittest.TestCase):
    """Test ensure_expr helper function."""

    def test_ensure_expr_with_lazy_expr(self):
        """Test ensure_expr with LazyStateDictExpr."""
        leaf = StateDictLeaf(create_simple_state_dict())
        result = ensure_expr(leaf)

        self.assertIs(result, leaf)

    def test_ensure_expr_with_dict(self):
        """Test ensure_expr with regular dict."""
        state_dict = create_simple_state_dict()
        result = ensure_expr(state_dict)

        self.assertIsInstance(result, StateDictLeaf)
        self.assertTrue(torch.equal(result["weight"], state_dict["weight"]))

    def test_ensure_expr_with_ordered_dict(self):
        """Test ensure_expr with OrderedDict."""
        state_dict = create_simple_state_dict()
        result = ensure_expr(state_dict)

        self.assertIsInstance(result, StateDictLeaf)

    def test_ensure_expr_with_invalid_type(self):
        """Test ensure_expr with invalid type."""
        with self.assertRaises(TypeError):
            ensure_expr("not a valid type")

        with self.assertRaises(TypeError):
            ensure_expr(123)

        with self.assertRaises(TypeError):
            ensure_expr([1, 2, 3])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_operations_preserve_tensor_properties(self):
        """Test that operations preserve tensor properties."""
        # Test with different dtypes
        state_dict_float32 = OrderedDict(
            [
                ("weight", torch.tensor([1.0, 2.0], dtype=torch.float32)),
                ("bias", torch.tensor([0.1], dtype=torch.float32)),
            ]
        )
        leaf = StateDictLeaf(state_dict_float32)
        result = (leaf * 2.0).materialize()

        for tensor in result.values():
            self.assertEqual(tensor.dtype, torch.float32)

    def test_operations_with_requires_grad(self):
        """Test operations with tensors requiring gradients."""
        state_dict_grad = OrderedDict(
            [
                ("weight", torch.tensor([1.0, 2.0], requires_grad=True)),
                ("bias", torch.tensor([0.1], requires_grad=True)),
            ]
        )
        leaf = StateDictLeaf(state_dict_grad)
        result = (leaf * 2.0 + leaf)["weight"]

        # Result should support gradients
        self.assertTrue(result.requires_grad)

    def test_empty_expression_chain(self):
        """Test expression with empty state dict."""
        empty_dict = OrderedDict()
        leaf = StateDictLeaf(empty_dict)

        result = (leaf * 2.0).materialize()
        self.assertEqual(len(result), 0)

    def test_single_key_state_dict(self):
        """Test with single key state dict."""
        single_key = OrderedDict([("weight", torch.tensor([1.0, 2.0]))])
        leaf = StateDictLeaf(single_key)

        result = (leaf * 3.0).materialize()
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result["weight"], torch.tensor([3.0, 6.0])))

    def test_repr_method(self):
        """Test __repr__ method returns meaningful string."""
        leaf = StateDictLeaf(create_simple_state_dict())
        repr_str = repr(leaf)

        self.assertIsInstance(repr_str, str)
        self.assertIn("StateDictLeaf", repr_str)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of lazy evaluation."""

    def test_lazy_evaluation_no_premature_computation(self):
        """Verify that operations don't compute until accessed."""
        # Create a large state dict (but don't access it yet)
        large_state_dict = OrderedDict(
            [("param1", torch.randn(1000, 1000)), ("param2", torch.randn(1000, 1000))]
        )

        leaf = StateDictLeaf(large_state_dict)

        # Build complex expression without accessing
        expr = (leaf + leaf) * 2.0 - leaf / 3.0

        # At this point, no actual computation should have happened
        # (We can't directly test this, but the test documents expected behavior)

        # Now access one key
        result = expr["param1"]
        self.assertIsInstance(result, torch.Tensor)

    def test_partial_materialization(self):
        """Test that we can access specific keys without computing all."""
        state_dict = OrderedDict(
            [
                ("key1", torch.tensor([1.0])),
                ("key2", torch.tensor([2.0])),
                ("key3", torch.tensor([3.0])),
            ]
        )

        leaf = StateDictLeaf(state_dict)
        expr = leaf * 2.0

        # Access only one key
        result = expr["key1"]
        self.assertTrue(torch.equal(result, torch.tensor([2.0])))

        # This documents that lazy evaluation allows partial computation


if __name__ == "__main__":
    unittest.main()
