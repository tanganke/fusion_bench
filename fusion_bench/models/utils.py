from typing import List

import torch
from torch import nn

from fusion_bench.utils.type import StateDictType


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def check_parameterNamesMatch(checkpoints: List[StateDictType]) -> None:
    """
    Checks that the parameter names of the given checkpoints match.

    Args:
        checkpoints (List[Dict[str, float]]): A list of checkpoints, where each checkpoint is a dictionary of parameter names and their corresponding values.

    Raises:
        ValueError: If the number of checkpoints is less than 2 or if the parameter names of any two checkpoints differ.

    """
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def find_layers_with_type(
    module: nn.Module,
    layer_types=[nn.Linear],
    prefix="",
):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layer_types (list): List of layer types to find.
        prefix (str): A prefix to add to the layer names.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    res = {}
    for name, submodule in module.named_modules(prefix=prefix):
        if isinstance(submodule, tuple(layer_types)):
            res[name] = submodule
    return res


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
