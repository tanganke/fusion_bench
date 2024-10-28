from typing import Dict, Union

import torch
from torch import Tensor, nn

from fusion_bench.utils.type import StateDictType


def param_random_drop_(param: Tensor, sparsity_level: float, rescale: bool):
    """
    Randomly drops elements in the given tensor based on the sparsity level.

    Args:
        param (Tensor): The tensor whose elements are to be randomly dropped.
        sparsity_level (float): The fraction of elements to drop (between 0 and 1).
        rescale (bool): If True, rescale the remaining elements to maintain the original sum.

    Returns:
        None
    """
    mask = torch.rand_like(param) > sparsity_level
    param.data = param.data * mask
    if rescale:
        param.data = param.data / (1 - sparsity_level)


def module_random_drop_(
    tv: Union[nn.Module, StateDictType], sparsity_level: float, rescale: bool
):
    """
    Applies random drop to all parameters in a module or state dictionary.

    Args:
        tv (Union[nn.Module, StateDictType]): The module or state dictionary whose parameters are to be randomly dropped.
        sparsity_level (float): The fraction of elements to drop (between 0 and 1).
        rescale (bool): If True, rescale the remaining elements to maintain the original sum.

    Returns:
        None
    """
    if isinstance(tv, nn.Module):
        for param in tv.parameters():
            param_random_drop_(param, sparsity_level, rescale)
    else:
        for param in tv.values():
            param_random_drop_(param, sparsity_level, rescale)


def trainable_state_dict(module: nn.Module):
    """
    Returns a state dictionary containing only the trainable parameters of the given module.

    Args:
        module (nn.Module): The module from which to extract the trainable parameters.

    Returns:
        dict: A dictionary where the keys are parameter names and the values are the corresponding trainable parameters.
    """
    return {
        name: param for name, param in module.named_parameters() if param.requires_grad
    }


def module_sub_(
    a: Union[nn.Module, StateDictType],
    b: Union[nn.Module, StateDictType],
    trainable_only: bool = True,
):
    """
    Subtracts the parameters of module b from module a in-place.

    Args:
        a (nn.Module): The module whose parameters will be subtracted from.
        b (nn.Module): The module whose parameters will be subtracted.

    Returns:
        nn.Module: The modified module a with updated parameters.
    """
    for (a_name, a_param), (b_name, b_param) in zip(
        a.named_parameters() if isinstance(a, nn.Module) else a.items(),
        b.named_parameters() if isinstance(b, nn.Module) else b.items(),
    ):
        assert a_name == b_name, "Mismatch in parameter names"
        if trainable_only and not a_param.requires_grad:
            continue
        a_param.data = a_param.data - b_param.data
    return a
