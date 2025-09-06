from collections import OrderedDict
from numbers import Number
from typing import Callable, Dict, List, Literal, Optional, Union, cast

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .parameters import check_parameters_all_equal
from .type import BoolStateDictType, StateDictType


def _validate_state_dict_list_not_empty(state_dicts: List[StateDictType]) -> None:
    """
    Validate that the list of state dicts is not empty and contains valid state dicts.

    Args:
        state_dicts: List of state dictionaries to validate.

    Raises:
        TypeError: If state_dicts is not a list or contains non-dict items.
        ValueError: If the list is empty or contains empty state dicts.
    """
    if state_dicts is None:
        raise TypeError("state_dicts cannot be None")

    if not isinstance(state_dicts, (list, tuple)):
        raise TypeError(
            f"Expected list or tuple of state dicts, got {type(state_dicts).__name__}"
        )

    if not state_dicts:
        raise ValueError("The list of state_dicts must not be empty")

    for i, state_dict in enumerate(state_dicts):
        if state_dict is None:
            raise ValueError(f"State dict at index {i} is None")
        if not isinstance(state_dict, (dict, OrderedDict)):
            raise TypeError(
                f"Item at index {i} is not a dictionary, got {type(state_dict).__name__}"
            )
        if not state_dict:
            raise ValueError(f"State dict at index {i} is empty")


def _validate_state_dict_same_keys(state_dicts: List[StateDictType]) -> None:
    """
    Validate that all state dicts have the same keys and compatible tensor shapes.

    Args:
        state_dicts: List of state dictionaries to validate.

    Raises:
        ValueError: If state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    if not state_dicts:
        return

    if len(state_dicts) < 2:
        return

    reference_state_dict = state_dicts[0]
    reference_keys = set(reference_state_dict.keys())

    if not reference_keys:
        raise ValueError("Reference state dict (index 0) has no keys")

    for i, state_dict in enumerate(state_dicts[1:], 1):
        current_keys = set(state_dict.keys())

        # Check for missing keys
        missing_keys = reference_keys - current_keys
        if missing_keys:
            raise ValueError(
                f"State dict at index {i} is missing keys: {sorted(missing_keys)}"
            )

        # Check for extra keys
        extra_keys = current_keys - reference_keys
        if extra_keys:
            raise ValueError(
                f"State dict at index {i} has extra keys: {sorted(extra_keys)}"
            )

        # Check tensor shapes and dtypes for compatibility
        for key in reference_keys:
            ref_tensor = reference_state_dict[key]
            curr_tensor = state_dict[key]

            # Handle None values
            if ref_tensor is None and curr_tensor is None:
                continue
            if ref_tensor is None or curr_tensor is None:
                raise ValueError(
                    f"Tensor None mismatch for key '{key}' at index {i}: "
                    f"one is None, the other is not"
                )

            if not isinstance(curr_tensor, type(ref_tensor)):
                raise TypeError(
                    f"Tensor type mismatch for key '{key}' at index {i}: "
                    f"expected {type(ref_tensor).__name__}, got {type(curr_tensor).__name__}"
                )

            if hasattr(ref_tensor, "shape") and hasattr(curr_tensor, "shape"):
                if ref_tensor.shape != curr_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for key '{key}' at index {i}: "
                        f"expected {ref_tensor.shape}, got {curr_tensor.shape}"
                    )

                if hasattr(ref_tensor, "dtype") and hasattr(curr_tensor, "dtype"):
                    if ref_tensor.dtype != curr_tensor.dtype:
                        raise ValueError(
                            f"Dtype mismatch for key '{key}' at index {i}: "
                            f"expected {ref_tensor.dtype}, got {curr_tensor.dtype}"
                        )

                # Check device compatibility (warn but don't fail)
                if (
                    hasattr(ref_tensor, "device")
                    and hasattr(curr_tensor, "device")
                    and ref_tensor.device != curr_tensor.device
                ):
                    import warnings

                    warnings.warn(
                        f"Device mismatch for key '{key}' at index {i}: "
                        f"reference on {ref_tensor.device}, current on {curr_tensor.device}. "
                        f"This may cause issues during arithmetic operations."
                    )


def _validate_list_lengths_equal(
    list1: List,
    list2: List,
    name1: str = "the first list",
    name2: str = "the second list",
) -> None:
    """
    Validate that two lists have the same length and are valid.

    Args:
        list1: First list to compare.
        list2: Second list to compare.
        name1: Descriptive name for the first list.
        name2: Descriptive name for the second list.

    Raises:
        TypeError: If either argument is not a list or names are not strings.
        ValueError: If the lists have different lengths or are empty.
    """
    # Validate input types
    if not isinstance(name1, str) or not isinstance(name2, str):
        raise TypeError("List names must be strings")

    if list1 is None or list2 is None:
        raise TypeError("Lists cannot be None")

    if not isinstance(list1, (list, tuple)):
        raise TypeError(f"{name1} must be a list or tuple, got {type(list1).__name__}")
    if not isinstance(list2, (list, tuple)):
        raise TypeError(f"{name2} must be a list or tuple, got {type(list2).__name__}")

    if not list1 and not list2:
        raise ValueError(f"Both {name1} and {name2} are empty")

    len1, len2 = len(list1), len(list2)
    if len1 != len2:
        raise ValueError(
            f"Length mismatch: {name1} has {len1} items, " f"{name2} has {len2} items"
        )

    # Additional validation for numeric lists (common use case)
    if list1 and hasattr(list1[0], "__float__"):  # Likely numeric
        try:
            # Check for NaN or infinite values in numeric lists
            import math

            for i, val in enumerate(list1):
                if isinstance(val, (int, float)) and (
                    math.isnan(val) or math.isinf(val)
                ):
                    raise ValueError(
                        f"{name1} contains invalid numeric value at index {i}: {val}"
                    )
            for i, val in enumerate(list2):
                if isinstance(val, (int, float)) and (
                    math.isnan(val) or math.isinf(val)
                ):
                    raise ValueError(
                        f"{name2} contains invalid numeric value at index {i}: {val}"
                    )
        except (TypeError, AttributeError):
            # If we can't check numeric values, skip this validation
            pass


def to_device(
    state_dict: StateDictType,
    device: Union[torch.device, str],
    copy: bool = False,
    inplace: bool = False,
) -> StateDictType:
    """
    Move state dict tensors to the specified device.

    Args:
        state_dict: The state dictionary to move.
        device: Target device for the tensors.
        copy: Whether to force a copy even when the tensor is already on the target device.
        inplace: Whether to modify the input state dict in place.

    Returns:
        State dict with tensors moved to the specified device.
    """
    if inplace:
        ret_state_dict = state_dict
    else:
        ret_state_dict = OrderedDict()

    for key, tensor in state_dict.items():
        ret_state_dict[key] = cast(Tensor, tensor).to(
            device, non_blocking=True, copy=copy
        )
    return ret_state_dict


def state_dicts_check_keys(state_dicts: List[StateDictType]) -> None:
    """
    Check that all state dictionaries have the same keys.

    Args:
        state_dicts: A list of state dictionaries to check.

    Raises:
        ValueError: If the state dictionaries have different keys or the list is empty.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_state_dict_same_keys(state_dicts)


def num_params_of_state_dict(state_dict: StateDictType) -> int:
    """
    Calculate the total number of parameters in a state dict.

    Args:
        state_dict: The state dict to count parameters in.

    Returns:
        The total number of parameters in the state dict.
    """
    return sum(tensor.numel() for tensor in state_dict.values())


def state_dict_flatten(state_dict: StateDictType) -> Tensor:
    """
    Flatten all tensors in a state dict into a single 1D tensor.

    Args:
        state_dict: The state dict to flatten.

    Returns:
        A single flattened tensor containing all parameters.
    """
    return torch.cat([tensor.flatten() for tensor in state_dict.values()])


def state_dict_avg(state_dicts: List[StateDictType]) -> StateDictType:
    """
    Calculate the element-wise average of a list of state dicts.

    Args:
        state_dicts: List of state dicts to average.

    Returns:
        A state dict containing the averaged parameters.

    Raises:
        ValueError: If the list is empty or state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_state_dict_same_keys(state_dicts)

    num_state_dicts = len(state_dicts)
    avg_state_dict = OrderedDict()

    # Initialize with zeros_like for better performance
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key])

    # Accumulate all state dicts
    for state_dict in state_dicts:
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # Divide by number of state dicts
    for key in avg_state_dict:
        avg_state_dict[key] /= num_state_dicts

    return avg_state_dict


def state_dict_sub(
    a: StateDictType,
    b: StateDictType,
    strict: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> StateDictType:
    """
    Compute the element-wise difference between two state dicts (a - b).

    Args:
        a: The first state dict (minuend).
        b: The second state dict (subtrahend).
        strict: Whether to require exact key matching between state dicts.
        device: Optional device to move the result tensors to.

    Returns:
        A state dict containing the element-wise differences.

    Raises:
        ValueError: If strict=True and the state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    result = OrderedDict()

    if strict:
        _validate_state_dict_same_keys([a, b])
        keys_to_process = a.keys()
    else:
        keys_to_process = set(a.keys()) & set(b.keys())

    for key in keys_to_process:
        result_tensor = a[key] - b[key]
        if device is not None:
            result_tensor = result_tensor.to(device, non_blocking=True)
        result[key] = result_tensor

    return result


def state_dict_add(
    a: StateDictType,
    b: StateDictType,
    strict: bool = True,
    device: Optional[Union[torch.device, str]] = None,
    show_pbar: bool = False,
) -> StateDictType:
    """
    Compute the element-wise sum of two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.
        strict: Whether to require exact key matching between state dicts.
        device: Optional device to move the result tensors to.
        show_pbar: Whether to show a progress bar during computation.

    Returns:
        A state dict containing the element-wise sums.

    Raises:
        ValueError: If strict=True and the state dicts have different parameters.
    """
    result = OrderedDict()

    if strict:
        _validate_state_dict_same_keys([a, b])
        keys_to_process = a.keys()
    else:
        keys_to_process = set(a.keys()) & set(b.keys())

    keys_iter = (
        tqdm(keys_to_process, desc="Adding state dicts")
        if show_pbar
        else keys_to_process
    )

    for key in keys_iter:
        if key in b:  # This check is redundant when strict=True but harmless
            result[key] = a[key] + b[key]

    if device is not None:
        result = to_device(result, device)

    return result


def state_dict_add_scalar(state_dict: StateDictType, scalar: Number) -> StateDictType:
    """
    Add a scalar value to all parameters in a state dict.

    Args:
        state_dict: The state dict to modify.
        scalar: The scalar value to add to each parameter.

    Returns:
        A new state dict with the scalar added to each parameter.
    """
    return OrderedDict((key, tensor + scalar) for key, tensor in state_dict.items())


def state_dict_mul(state_dict: StateDictType, scalar: float) -> StateDictType:
    """
    Multiply all parameters in a state dict by a scalar.

    Args:
        state_dict: The state dict to multiply.
        scalar: The scalar value to multiply each parameter by.

    Returns:
        A new state dict with each parameter multiplied by the scalar.
    """
    return OrderedDict((key, scalar * tensor) for key, tensor in state_dict.items())


def state_dict_div(
    state_dict: StateDictType, scalar: float, show_pbar: bool = False
) -> StateDictType:
    """
    Divide all parameters in a state dict by a scalar.

    Args:
        state_dict: The state dict to divide.
        scalar: The scalar value to divide each parameter by.
        show_pbar: Whether to show a progress bar during computation.

    Returns:
        A new state dict with each parameter divided by the scalar.

    Raises:
        ZeroDivisionError: If scalar is zero.
    """
    if scalar == 0:
        raise ZeroDivisionError("Cannot divide state dict by zero")

    keys_iter = (
        tqdm(state_dict.keys(), desc="Dividing state dict")
        if show_pbar
        else state_dict.keys()
    )
    return OrderedDict((key, state_dict[key] / scalar) for key in keys_iter)


def state_dict_power(state_dict: StateDictType, p: float) -> StateDictType:
    """
    Raise all parameters in a state dict to a power.

    Args:
        state_dict: The state dict to raise to a power.
        p: The exponent to raise each parameter to.

    Returns:
        A new state dict with each parameter raised to the power p.
    """
    return OrderedDict((key, tensor**p) for key, tensor in state_dict.items())


def state_dict_interpolation(
    state_dicts: List[StateDictType], scalars: List[float]
) -> StateDictType:
    """
    Interpolate between multiple state dicts using specified scalar weights.

    Args:
        state_dicts: List of state dicts to interpolate between.
        scalars: List of scalar weights for interpolation.

    Returns:
        A state dict containing the interpolated parameters.

    Raises:
        ValueError: If the lists have different lengths or are empty, or if state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_list_lengths_equal(state_dicts, scalars, "state_dicts", "scalars")
    _validate_state_dict_same_keys(state_dicts)

    interpolated_state_dict = OrderedDict()

    # Initialize with zeros
    for key in state_dicts[0]:
        interpolated_state_dict[key] = torch.zeros_like(state_dicts[0][key])

    # Accumulate weighted contributions
    for state_dict, scalar in zip(state_dicts, scalars):
        for key in interpolated_state_dict:
            interpolated_state_dict[key] += scalar * state_dict[key]

    return interpolated_state_dict


def state_dict_sum(state_dicts: List[StateDictType]) -> StateDictType:
    """
    Compute the element-wise sum of multiple state dicts.

    Args:
        state_dicts: List of state dicts to sum.

    Returns:
        A state dict containing the element-wise sums.

    Raises:
        ValueError: If the list is empty or state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_state_dict_same_keys(state_dicts)

    sum_state_dict = OrderedDict()

    # Initialize with zeros
    for key in state_dicts[0]:
        sum_state_dict[key] = torch.zeros_like(state_dicts[0][key])

    # Accumulate all state dicts
    for state_dict in state_dicts:
        for key in sum_state_dict:
            sum_state_dict[key] += state_dict[key]

    return sum_state_dict


def state_dict_weighted_sum(
    state_dicts: List[StateDictType],
    weights: List[float],
    device: Optional[Union[torch.device, str]] = None,
) -> StateDictType:
    """
    Compute the weighted sum of multiple state dicts.

    Args:
        state_dicts: List of state dicts to combine.
        weights: List of weights for the weighted sum.
        device: Optional device to move the result tensors to.

    Returns:
        A state dict containing the weighted sum of parameters.

    Raises:
        ValueError: If the lists have different lengths or are empty, or if state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_list_lengths_equal(state_dicts, weights, "state_dicts", "weights")
    _validate_state_dict_same_keys(state_dicts)

    weighted_sum_state_dict = OrderedDict()

    # Single pass initialization and computation for better performance
    for key in state_dicts[0]:
        # Get reference tensor and handle sparse tensors
        ref_tensor = state_dicts[0][key]
        is_sparse = ref_tensor.is_sparse if hasattr(ref_tensor, "is_sparse") else False

        # Initialize result tensor
        if is_sparse:
            # For sparse tensors, start with zeros in dense format for efficient accumulation
            result_tensor = torch.zeros_like(ref_tensor).to_dense()
        else:
            result_tensor = torch.zeros_like(ref_tensor)

        # Accumulate weighted contributions in a single loop
        for state_dict, weight in zip(state_dicts, weights):
            tensor = state_dict[key]

            # Optimize for common cases
            if weight == 0.0:
                continue  # Skip zero weights
            elif weight == 1.0:
                result_tensor += tensor  # Avoid multiplication for unit weights
            else:
                # Use in-place operations when possible for memory efficiency
                if is_sparse and hasattr(tensor, "is_sparse") and tensor.is_sparse:
                    result_tensor += weight * tensor.to_dense()
                else:
                    result_tensor += weight * tensor

        # Move to target device if specified (do this once per tensor, not per operation)
        if device is not None:
            result_tensor = result_tensor.to(device, non_blocking=True)

        # Convert back to sparse if original was sparse and result is suitable
        if is_sparse and hasattr(result_tensor, "to_sparse"):
            try:
                # Only convert back to sparse if it would be memory efficient
                # (i.e., if the result has sufficient sparsity)
                if result_tensor.numel() > 0:
                    sparsity_ratio = (result_tensor == 0).float().mean().item()
                    if sparsity_ratio > 0.5:  # Convert back if >50% zeros
                        result_tensor = result_tensor.to_sparse()
            except (RuntimeError, AttributeError):
                # If conversion fails, keep as dense
                pass

        weighted_sum_state_dict[key] = result_tensor

    return weighted_sum_state_dict


def state_dict_diff_abs(a: StateDictType, b: StateDictType) -> StateDictType:
    """
    Compute the element-wise absolute difference between two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.

    Returns:
        A state dict containing the absolute differences.
    """
    diff = state_dict_sub(a, b)
    return OrderedDict((key, tensor.abs()) for key, tensor in diff.items())


def state_dict_binary_mask(
    a: StateDictType,
    b: StateDictType,
    compare_fn: Union[
        Literal["greater", "less", "equal", "not_equal"],
        Callable[[Tensor, Tensor], torch.BoolTensor],
    ] = "greater",
    strict: bool = True,
    show_pbar: bool = False,
) -> BoolStateDictType:
    """
    Create binary masks by comparing elements in two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.
        compare_fn: Comparison function to use. Can be a string literal
                   ("greater", "less", "equal", "not_equal") or a callable
                   that takes two tensors and returns a boolean tensor.
        strict: Whether to require exact key matching between state dicts.
        show_pbar: Whether to show a progress bar during computation.

    Returns:
        A dictionary containing boolean masks based on the comparison.

    Raises:
        ValueError: If compare_fn is not a valid string or callable, or if strict=True
                   and the state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    compare_fn_dict = {
        "greater": lambda x, y: x > y,
        "less": lambda x, y: x < y,
        "equal": lambda x, y: x == y,
        "not_equal": lambda x, y: x != y,
    }

    if isinstance(compare_fn, str):
        if compare_fn not in compare_fn_dict:
            raise ValueError(
                f"Invalid compare_fn string: {compare_fn}. Must be one of {list(compare_fn_dict.keys())}"
            )
        compare_fn = compare_fn_dict[compare_fn]
    elif not callable(compare_fn):
        raise ValueError(
            f"compare_fn must be a string or a callable, but got {type(compare_fn)}"
        )

    result = OrderedDict()

    if strict:
        _validate_state_dict_same_keys([a, b])
        keys_to_process = a.keys()
    else:
        keys_to_process = set(a.keys()) & set(b.keys())

    keys_iter = (
        tqdm(keys_to_process, desc="Creating binary masks")
        if show_pbar
        else keys_to_process
    )

    for key in keys_iter:
        result[key] = compare_fn(a[key], b[key])

    return result


def state_dict_hadamard_product(a: StateDictType, b: StateDictType) -> StateDictType:
    """
    Compute the Hadamard product (element-wise multiplication) of two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.

    Returns:
        A state dict containing the element-wise products.

    Raises:
        ValueError: If the state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    _validate_state_dict_same_keys([a, b])
    return OrderedDict((key, a[key] * b[key]) for key in a)
