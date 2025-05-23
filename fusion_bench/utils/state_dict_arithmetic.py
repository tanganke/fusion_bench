from collections import OrderedDict
from numbers import Number
from typing import Callable, Dict, List, Literal, Union, cast

import torch
from torch import Tensor

from .parameters import check_parameters_all_equal
from .type import BoolStateDictType, StateDictType


def to_device(
    state_dict: StateDictType,
    device: Union[torch.device, str],
    copy: bool = False,
    inplace: bool = False,
):
    if inplace:
        ret_state_dict = state_dict
    else:
        ret_state_dict = OrderedDict()
    for key in state_dict:
        ret_state_dict[key] = cast(Tensor, state_dict[key]).to(
            device, non_blocking=True, copy=copy
        )
    return ret_state_dict


def state_dicts_check_keys(state_dicts: List[StateDictType]):
    """
    Checks that the state dictionaries have the same keys.

    Args:
        state_dicts (List[Dict[str, Tensor]]): A list of dictionaries containing the state of PyTorch models.

    Raises:
        ValueError: If the state dictionaries have different keys.
    """
    # Get the keys of the first state dictionary in the list
    keys = set(state_dicts[0].keys())
    # Check that all the state dictionaries have the same keys
    for state_dict in state_dicts:
        assert keys == set(state_dict.keys()), "keys of state_dicts are not equal"


def num_params_of_state_dict(state_dict: StateDictType):
    """
    Returns the number of parameters in a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to count the number of parameters in.

    Returns:
        int: The number of parameters in the state dict.
    """
    return sum([state_dict[key].numel() for key in state_dict])


def state_dict_flatten(state_dict: Dict[str, Tensor]):
    """
    Flattens a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to be flattened.

    Returns:
        Tensor: The flattened state dict.
    """
    flattened_state_dict = []
    for key in state_dict:
        flattened_state_dict.append(state_dict[key].flatten())
    return torch.cat(flattened_state_dict)


def state_dict_avg(state_dicts: List[StateDictType]):
    """
    Returns the average of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to average.

    Returns:
        Dict: The average of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all(
        [len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]
    ), "All state_dicts must have the same number of keys"

    num_state_dicts = len(state_dicts)
    avg_state_dict = OrderedDict()
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            avg_state_dict[key] += state_dict[key]
        avg_state_dict[key] /= num_state_dicts
    return avg_state_dict


def state_dict_sub(
    a: StateDictType, b: StateDictType, strict: bool = True, device=None
):
    """
    Returns the difference between two state dicts `a-b`.

    Args:
        a (StateDictType): The first state dict.
        b (StateDictType): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        StateDictType: The difference between the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = OrderedDict()
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
            if device is not None:
                diff[k] = diff[k].to(device, non_blocking=True)
    return diff


def state_dict_add(
    a: StateDictType, b: StateDictType, strict: bool = True, device=None
):
    """
    Returns the sum of two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The sum of the two state dicts.
    """
    ans = {}
    if strict:
        check_parameters_all_equal([a, b])
        for key in a:
            ans[key] = a[key] + b[key]
    else:
        for key in a:
            if key in b:
                ans[key] = a[key] + b[key]
    if device is not None:
        ans = to_device(ans, device)
    return ans


def state_dict_add_scalar(a: StateDictType, scalar: Number):
    ans = OrderedDict()
    for key in a:
        ans[key] = a[key] + scalar
    return ans


def state_dict_mul(state_dict: StateDictType, scalar: float):
    """
    Returns the product of a state dict and a scalar.

    Args:
        state_dict (Dict): The state dict to be multiplied.
        scalar (float): The scalar to multiply the state dict with.

    Returns:
        Dict: The product of the state dict and the scalar.
    """
    diff = OrderedDict()
    for k in state_dict:
        diff[k] = scalar * state_dict[k]
    return diff


def state_dict_div(state_dict: StateDictType, scalar: float):
    """
    Returns the division of a state dict by a scalar.

    Args:
        state_dict (Dict): The state dict to be divided.
        scalar (float): The scalar to divide the state dict by.

    Returns:
        Dict: The division of the state dict by the scalar.
    """
    diff = OrderedDict()
    for k in state_dict:
        diff[k] = state_dict[k] / scalar
    return diff


def state_dict_power(state_dict: Dict[str, Tensor], p: float):
    """
    Returns the power of a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to be powered.
        p (float): The power to raise the state dict to.

    Returns:
        Dict[str, Tensor]: The powered state dict.
    """
    powered_state_dict = {}
    for key in state_dict:
        powered_state_dict[key] = state_dict[key] ** p
    return powered_state_dict


def state_dict_interpolation(
    state_dicts: List[Dict[str, Tensor]], scalars: List[float]
):
    """
    Interpolates between a list of state dicts using a list of scalars.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        scalars (List[float]): The list of scalars to use for interpolation.

    Returns:
        Dict: The interpolated state dict.
    """
    assert len(state_dicts) == len(
        scalars
    ), "The number of state_dicts and scalars must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all(
        [len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]
    ), "All state_dicts must have the same number of keys"

    interpolated_state_dict = {}
    for key in state_dicts[0]:
        interpolated_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict, scalar in zip(state_dicts, scalars):
            interpolated_state_dict[key] += scalar * state_dict[key]
    return interpolated_state_dict


def state_dict_sum(state_dicts: List[StateDictType]):
    """
    Returns the sum of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to sum.

    Returns:
        Dict: The sum of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all(
        [len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]
    ), "All state_dicts must have the same number of keys"

    sum_state_dict = OrderedDict()
    for key in state_dicts[0]:
        sum_state_dict[key] = 0
        for state_dict in state_dicts:
            sum_state_dict[key] = sum_state_dict[key] + state_dict[key]
    return sum_state_dict


def state_dict_weighted_sum(
    state_dicts: List[Dict[str, Tensor]], weights: List[float], device=None
):
    """
    Returns the weighted sum of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        weights (List[float]): The list of weights to use for the weighted sum.

    Returns:
        Dict: The weighted sum of the state dicts.
    """
    assert len(state_dicts) == len(
        weights
    ), "The number of state_dicts and weights must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all(
        [len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]
    ), "All state_dicts must have the same number of keys"

    weighted_sum_state_dict: Dict[str, Tensor] = {}
    for key in state_dicts[0]:
        # states dicts can be sparse matrices
        weighted_sum_state_dict[key] = torch.zeros_like(state_dicts[0][key]).to_dense()
        for state_dict, weight in zip(state_dicts, weights):
            weighted_sum_state_dict[key] = torch.add(
                weighted_sum_state_dict[key], weight * state_dict[key]
            )
        if device is not None:
            weighted_sum_state_dict[key] = weighted_sum_state_dict[key].to(
                device, non_blocking=True
            )
    return weighted_sum_state_dict


def state_dict_diff_abs(a: StateDictType, b: StateDictType):
    """
    Returns the per-layer abs of the difference between two state dicts.

    Args:
        a (StateDictType): The first state dict.
        b (StateDictType): The second state dict.

    Returns:
        StateDictType: The absolute difference between the two state dicts.
    """
    diff = state_dict_sub(a, b)
    abs_diff = {key: diff[key].abs() for key in diff}
    return abs_diff


def state_dict_binary_mask(
    a: StateDictType,
    b: StateDictType,
    compare_fn: Union[
        Literal["greater", "less", "equal", "not_equal"],
        Callable[[Tensor, Tensor], torch.BoolTensor],
    ] = "greater",
) -> BoolStateDictType:
    """
    Returns the binary mask of elements in a compared to elements in b using the provided comparison function.

    Args:
        a (StateDictType): The first state dict.
        b (StateDictType): The second state dict.
        compare_fn (Union[Literal["greater", "less", "equal", "not_equal"], Callable[[Tensor, Tensor], Tensor]]): A function that takes two tensors and returns a boolean tensor.
            Defaults to greater than comparison (x > y).

    Returns:
        StateDictType: A dictionary containing binary masks (0 or 1) based on the comparison.
    """
    compare_fn_dict = {
        "greater": lambda x, y: x > y,
        "less": lambda x, y: x < y,
        "equal": lambda x, y: x == y,
        "not_equal": lambda x, y: x != y,
    }
    if isinstance(compare_fn, str):
        compare_fn = compare_fn_dict[compare_fn]
    elif not callable(compare_fn):
        raise ValueError(
            f"compare_fn must be a string or a callable, but got {type(compare_fn)}"
        )

    mask = OrderedDict()
    for key in a:
        mask[key] = compare_fn(a[key], b[key])
    return mask


def state_dict_hadmard_product(a: StateDictType, b: StateDictType) -> StateDictType:
    """
    Returns the Hadamard product of two state dicts, i.e. element-wise product.

    Args:
        a (StateDictType): The first state dict.
        b (StateDictType): The second state dict.

    Returns:
        StateDictType: The Hadamard product of the two state dicts.
    """
    ans = OrderedDict()
    for key in a:
        ans[key] = a[key] * b[key]
    return ans
