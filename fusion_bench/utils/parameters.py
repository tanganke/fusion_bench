import logging
from typing import Callable, Iterable, List, Mapping, Optional, Union

import torch
from torch import nn

from .type import _StateDict

__all__ = ["count_parameters", "print_parameters", "check_parameters_all_equal"]


def human_readable(num: int) -> str:
    if num < 1000 and isinstance(num, int):
        return str(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "B", "T", "P"][magnitude])


@torch.no_grad()
def count_parameters(module: nn.Module):
    """
    Counts the number of trainable and total parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model for which to count parameters.

    Returns:
        tuple: A tuple containing the number of trainable parameters and the total number of parameters.

    Examples:

        ```python
        # Count the parameters
        trainable_params, all_params = count_parameters(model)
        ```
    """
    trainable_params = 0
    all_param = 0
    for name, param in module.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_param


def print_parameters(
    module: nn.Module,
    human_readable: bool = True,
):
    """
    Prints the number of trainable and total parameters in a PyTorch model.

    Args:
        module (nn.Module): The PyTorch model for which to print parameters.
        human_readable (bool, optional): If True, the parameter counts are converted to a human-readable format (e.g., '1.5M' instead of '1500000'). Defaults to True.

    Prints:
        The number of trainable parameters, the total number of parameters, and the percentage of trainable parameters in the model.
    """
    trainable_params, all_param = count_parameters(module)
    if human_readable:
        trainable_params = human_readable(trainable_params)
        all_param = human_readable(all_param)

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def check_parameters_all_equal(
    list_of_param_names: List[Union[_StateDict, nn.Module, List[str]]]
) -> None:
    """
    Checks if all models have the same parameters.

    This function takes a list of parameter names or state dictionaries from different models.
    It checks if all models have the same parameters by comparing the parameter names.
    If any model has different parameters, it raises a ValueError with the differing parameters.

    Args:
        list_of_param_names (List[Union[StateDict, List[str]]]): A list of parameter names or state dictionaries.

    Raises:
        ValueError: If any model has different parameters.

    Returns:
        None
    """
    if isinstance(list_of_param_names[0], Mapping):
        list_of_param_names = [list(i.keys()) for i in list_of_param_names]
    elif isinstance(list_of_param_names[0], nn.Module):
        list_of_param_names = [list(i.state_dict().keys()) for i in list_of_param_names]
    else:
        parameter_names = set(list_of_param_names[0])

        if len(list_of_param_names) >= 2:
            # raise ValueError("Number of models is less than 2.")
            for names in list_of_param_names[1:]:
                current_parameterNames = set(names)
                if current_parameterNames != parameter_names:
                    raise ValueError(
                        "Differing parameter names in models. "
                        f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                    )
