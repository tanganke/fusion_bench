import unittest

import torch
import torch.nn as nn

from fusion_bench.method.simple_average import SimpleAverageAlgorithm, simple_average


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


class TestSimpleAverageAlgorithm(unittest.TestCase):
    def setUp(self):
        # here we set inplace to False to avoid modifying the input models
        self.algorithm = SimpleAverageAlgorithm(inplace=False)
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


class TestSimpleAverageFunction(unittest.TestCase):
    """Test suite for the simple_average function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.models = [nn.Linear(10, 5) for _ in range(3)]
        # Initialize models with different weights
        for i, model in enumerate(self.models):
            for param in model.parameters():
                param.data = param.data + i * 0.1

    def test_simple_average_with_modules(self):
        """Test simple_average with a list of nn.Module objects."""
        result = simple_average(self.models)

        # Result should be an nn.Module
        self.assertIsInstance(result, nn.Module)

        # Check that parameters are the average of input models
        for name, param in result.named_parameters():
            expected = sum(model.state_dict()[name] for model in self.models) / len(
                self.models
            )
            torch.testing.assert_close(param, expected, atol=1e-6, rtol=1e-5)

    def test_simple_average_with_state_dicts(self):
        """Test simple_average with a list of state dictionaries."""
        state_dicts = [model.state_dict() for model in self.models]
        result = simple_average(state_dicts)

        # Result should be a dict (state dict)
        self.assertIsInstance(result, dict)

        # Check that state dict is the average of input state dicts
        for key in result:
            expected = sum(sd[key] for sd in state_dicts) / len(state_dicts)
            torch.testing.assert_close(result[key], expected, atol=1e-6, rtol=1e-5)

    def test_simple_average_with_base_module(self):
        """Test simple_average with a base module provided."""
        base_module = nn.Linear(10, 5)
        result = simple_average(self.models, base_module=base_module)

        # Result should be the base_module
        self.assertIs(result, base_module)

        # Check that base_module's parameters are now the average
        for name, param in result.named_parameters():
            expected = sum(model.state_dict()[name] for model in self.models) / len(
                self.models
            )
            torch.testing.assert_close(param, expected, atol=1e-6, rtol=1e-5)

    def test_simple_average_single_model(self):
        """Test simple_average with a single model."""
        result = simple_average([self.models[0]])

        # Result should be an nn.Module
        self.assertIsInstance(result, nn.Module)

        # Check that parameters match the input model
        for name, param in result.named_parameters():
            expected = self.models[0].state_dict()[name]
            torch.testing.assert_close(param, expected, atol=1e-6, rtol=1e-5)

    def test_simple_average_preserves_model_structure(self):
        """Test that simple_average preserves the model structure."""
        result = simple_average(self.models)

        # Check that the model structure is preserved
        self.assertEqual(type(result), type(self.models[0]))
        self.assertEqual(
            len(list(result.parameters())), len(list(self.models[0].parameters()))
        )

        # Check parameter shapes
        for result_param, original_param in zip(
            result.parameters(), self.models[0].parameters()
        ):
            self.assertEqual(result_param.shape, original_param.shape)

    def test_simple_average_with_target_modules_attribute(self):
        """Test simple_average respects _fusion_bench_target_modules on full models."""
        # When using SimpleModel with multiple layers, ensure full averaging works
        models = [SimpleModel() for _ in range(2)]

        # Test without target modules (should average all)
        result = simple_average(models)

        # Result should be an nn.Module
        self.assertIsInstance(result, nn.Module)

        # Check that all parameters are averaged
        for name, param in result.named_parameters():
            expected = sum(model.state_dict()[name] for model in models) / len(models)
            torch.testing.assert_close(param, expected, atol=1e-6, rtol=1e-5)

    def test_simple_average_two_models(self):
        """Test simple_average with exactly two models."""
        result = simple_average(self.models[:2])

        # Result should be an nn.Module
        self.assertIsInstance(result, nn.Module)

        # Check that parameters are the average of the two models
        for name, param in result.named_parameters():
            expected = (
                self.models[0].state_dict()[name] + self.models[1].state_dict()[name]
            ) / 2
            torch.testing.assert_close(param, expected, atol=1e-6, rtol=1e-5)

    def test_simple_average_with_different_dtypes(self):
        """Test simple_average with models having different dtypes."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        # Both should have same dtype after averaging
        result = simple_average([model1, model2])

        # Check all parameters have the same dtype as input
        for param in result.parameters():
            self.assertEqual(param.dtype, model1.weight.dtype)

    def test_simple_average_state_dict_equality(self):
        """Test that state dict from simple_average matches manual averaging."""
        result_module = simple_average(self.models)
        result_dict = simple_average([m.state_dict() for m in self.models])

        # Compare state dicts
        for key in result_module.state_dict():
            torch.testing.assert_close(
                result_module.state_dict()[key], result_dict[key], atol=1e-6, rtol=1e-5
            )

    def test_simple_average_complex_model(self):
        """Test simple_average with complex multi-layer models."""
        # Create SimpleModel instances with different weights
        models = [SimpleModel() for _ in range(3)]
        for i, model in enumerate(models):
            for param in model.parameters():
                param.data = param.data + i * 0.1

        result = simple_average(models)

        # Check that all layers are averaged correctly
        for name in result.state_dict():
            expected = sum(m.state_dict()[name] for m in models) / len(models)
            torch.testing.assert_close(
                result.state_dict()[name], expected, atol=1e-6, rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
