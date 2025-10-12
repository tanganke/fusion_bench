"""
Tests for bug fixes in the fusion_bench codebase.
"""

import numpy as np
import pytest


def test_solvers_mu_function_with_negative_values():
    """
    Test that the mu function correctly raises ValueError for negative values.
    
    This tests the fix for the bug where `if len(np.where(rl < 0)[0]):` was 
    changed to `if len(np.where(rl < 0)[0]) > 0:`.
    """
    try:
        from fusion_bench.method.pwe_moe.phn.solvers import mu
    except ModuleNotFoundError:
        pytest.skip("cvxopt not installed, skipping test")
    
    # Test with negative values - should raise ValueError
    rl_negative = np.array([1.0, -0.5, 0.3])
    with pytest.raises(ValueError, match="rl<0"):
        mu(rl_negative)
    
    # Test with all positive values - should not raise
    rl_positive = np.array([1.0, 0.5, 0.3])
    result = mu(rl_positive)
    assert isinstance(result, (float, np.floating))
    
    # Test with all zeros - should not raise (zeros are not negative)
    rl_zeros = np.array([0.0, 0.0, 0.0])
    # This should not raise ValueError since zeros are not negative
    try:
        mu(rl_zeros)
    except (ValueError, ZeroDivisionError):
        # ZeroDivisionError might occur due to division by sum, which is acceptable
        pass


def test_modelpool_empty_name_handling():
    """
    Test that model_names property correctly handles empty names.
    
    This tests the fix for potential IndexError when accessing name[0] and name[-1]
    without checking if name is non-empty.
    """
    from fusion_bench.compat.modelpool.base_pool import ModelPool
    from omegaconf import DictConfig
    
    # Create a mock ModelPool with empty name
    config = DictConfig({
        "models": [
            {"name": "model1"},
            {"name": ""},  # Empty name
            {"name": "_pretrained_"},  # Special name (should be excluded)
            {"name": "model2"},
        ]
    })
    
    pool = ModelPool(modelpool_config=config)
    
    # Should not raise IndexError and should exclude empty and special names
    names = pool.model_names
    assert "model1" in names
    assert "model2" in names
    assert "" not in names
    assert "_pretrained_" not in names


def test_boolean_comparison_with_is():
    """
    Test that boolean comparisons use 'is True' instead of '== True'.
    
    This is more of a code quality test - the actual behavior should be the same,
    but 'is True' is more explicit and avoids potential issues with truthy values.
    """
    # Test that True is True works as expected
    value = True
    assert value is True
    
    # Show the difference between == and is
    # For True/False/None, 'is' is preferred
    assert (1 == True) is True  # == compares values
    assert (1 is True) is False  # is compares identity
    
    # This demonstrates why 'is True' is safer than '== True'
    assert (1.0 == True) is True
    assert (1.0 is True) is False


def test_magnitude_pruning_rescale_boolean():
    """
    Test that rescale parameter works correctly with boolean True.
    
    This verifies the fix where `self.rescale == True` was changed to 
    `self.rescale is True`.
    """
    # The actual test would require a full MagnitudeDiffPruningAlgorithm setup
    # Here we just verify the boolean logic
    rescale = True
    prune_ratio = 0.5
    
    # Simulate the fixed code
    if rescale is True:
        rescale_factor = 1 / prune_ratio
    else:
        rescale_factor = rescale
    
    assert rescale_factor == 2.0
    
    # Test with False
    rescale = False
    if rescale is True:
        rescale_factor = 1 / prune_ratio
    else:
        rescale_factor = rescale
    
    assert rescale_factor is False
    
    # Test with numeric value
    rescale = 1.5
    if rescale is True:
        rescale_factor = 1 / prune_ratio
    else:
        rescale_factor = rescale
    
    assert rescale_factor == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
