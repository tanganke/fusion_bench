import copy
from collections import OrderedDict
from typing import List, Mapping, Optional, Union

import torch
from torch import nn

from .type import StateDictType

__all__ = [
    "count_parameters",
    "print_parameters",
    "check_parameters_all_equal",
    "get_parameter_statistics",
    "state_dict_to_vector",
    "vector_to_state_dict",
    "trainable_state_dict",
]

# Model conversion utils


def trainable_state_dict(
    module: nn.Module,
    prefix: str = "",
    keep_vars: bool = False,
) -> StateDictType:
    """
    Returns the state dictionary of the module containing only the trainable parameters.

    Args:
        module (nn.Module): The neural network module.
        prefix (str, optional): The prefix to add to the parameter names. Defaults to "".
        keep_vars (bool, optional): If True, the parameters are not detached. Defaults to False.

    Returns:
        Dict[str, Tensor]: A dictionary containing the names and values of the trainable parameters.
    """
    state_dict = {
        prefix + name: param if keep_vars else param.detach()
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    return state_dict


def state_dict_to_vector(
    state_dict: Union[StateDictType, nn.Module],
    remove_keys: Optional[List[str]] = None,
):
    """
    Convert a state dictionary to a vector.

    Args:
        state_dict (Union[dict[str, torch.Tensor], nn.Module]): The state dictionary to convert.
        remove_keys (list, optional): List of keys to remove from the state dictionary. Defaults to [].

    Returns:
        torch.Tensor: The converted vector.
    """
    remove_keys = remove_keys if remove_keys is not None else []

    if isinstance(state_dict, nn.Module):
        shared_state_dict = state_dict.state_dict()
    else:
        shared_state_dict = copy.copy(state_dict)

    # remove the keys to be removed
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]

    # sort the reference dict
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))

    vector = nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )
    return vector


def vector_to_state_dict(
    vector: torch.Tensor,
    state_dict: Union[StateDictType, nn.Module],
    remove_keys: Optional[List[str]] = None,
):
    """
    Convert a vector to a state dictionary.

    Args:
        vector (torch.Tensor): The vector to convert.
        state_dict (Union[dict[str, torch.Tensor], nn.Module]): The reference state dictionary to define the order of the vector.
        remove_keys (list, optional): List of keys to remove from the reference state dictionary. Defaults to [].

    Returns:
        dict: The converted state dictionary.
    """
    remove_keys = remove_keys if remove_keys is not None else []

    # create a reference dict to define the order of the vector
    if isinstance(state_dict, nn.Module):
        reference_dict = state_dict.state_dict()
    else:
        # shallow copy the state_dict
        reference_dict = copy.copy(state_dict)

    # remove the keys to be removed
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]

    # sort the reference dict
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the reference dict
    nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def human_readable(num: int) -> str:
    """
    Converts a number into a human-readable string with appropriate magnitude suffix.

    Examples:

        ```python
        print(human_readable(1500))
        # Output: '1.50K'
        print(human_readable(1500000))
        # Output: '1.50M'
        ```

    Args:
        num (int): The number to convert.

    Returns:
        str: The human-readable string representation of the number.
    """
    if num < 1000 and isinstance(num, int):
        return str(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "B", "T", "P"][magnitude])


def _numel(param: torch.Tensor, non_zero_only: bool = False) -> int:
    """
    Count the number of elements in a tensor.

    Args:
        param (torch.Tensor): The tensor for which to count the number of elements.
        non_zero_only (bool, optional): If True, only non-zero elements are counted. If False, all elements are counted. Defaults to False.

    Returns:
        int: The number of elements in the tensor.
    """

    if non_zero_only:
        return torch.sum(param != 0).item()
    else:
        num_params = param.numel()

        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(
                param.quant_storage, "itemsize"
            ):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes
        return num_params


@torch.no_grad()
def count_parameters(module: nn.Module, non_zero_only: bool = False) -> tuple[int, int]:
    """
    Counts the number of trainable and total parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model for which to count parameters.
        non_zero_only (bool, optional): If True, only non-zero parameters are counted. If False, all parameters are counted. Defaults to False.

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
        # count the number of parameters
        num_params = _numel(param, non_zero_only)

        # accumulate the number of trainable and total parameters
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


@torch.no_grad()
def get_parameter_summary(
    module_or_state_dict: Union[nn.Module, StateDictType], non_zero_only: bool = False
) -> dict:
    """
    Get a summary of the parameters in a PyTorch model.
    """
    if isinstance(module_or_state_dict, nn.Module):
        state_dict = module_or_state_dict.state_dict(keep_vars=True)
    else:
        state_dict = module_or_state_dict

    trainable_params = 0
    all_param = 0
    bytes = 0

    for name, param in state_dict.items():
        # count the number of parameters
        num_params = _numel(param, non_zero_only)
        bytes += _numel(param, non_zero_only=False) * param.element_size()

        # accumulate the number of trainable and total parameters
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {
        "trainable_params": trainable_params,
        "all_param": all_param,
        "bytes": bytes,
    }


def print_parameters(
    module: nn.Module,
    is_human_readable: bool = True,
    print_fn=print,
    non_zero_only: bool = False,
):
    """
    Prints the number of trainable and total parameters in a PyTorch model.

    Args:
        module (nn.Module): The PyTorch model for which to print parameters.
        human_readable (bool, optional): If True, the parameter counts are converted to a human-readable format (e.g., '1.5M' instead of '1500000'). Defaults to True.
        print_fn (Callable): Function used to print the message.
        non_zero_only (bool, optional): If True, only non-zero elements are counted. If False, all elements are counted. Defaults to False.

    Prints:
        The number of trainable parameters, the total number of parameters, and the percentage of trainable parameters in the model.
    """
    trainable_params, all_param = count_parameters(module, non_zero_only=non_zero_only)
    trainable_ratio = 100 * trainable_params / all_param
    if is_human_readable:
        trainable_params = human_readable(trainable_params)
        all_param = human_readable(all_param)

    print_fn(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {trainable_ratio:.4f}"
    )


def check_parameters_all_equal(
    list_of_param_names: List[Union[StateDictType, nn.Module, List[str]]],
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


@torch.no_grad()
def get_parameter_statistics(
    module_or_state_dict: Union[nn.Module, StateDictType],
    model_wise: bool = False,
) -> dict:
    """
    Get statistics of the parameters in a PyTorch model or state dictionary.

    Args:
        module_or_state_dict (Union[nn.Module, StateDictType]): The PyTorch model for which to get parameter statistics.

    Returns:
        dict: A dictionary containing the mean, standard deviation, min, and max of the parameters.
    """
    stats = {}
    if isinstance(module_or_state_dict, nn.Module):
        state_dict = module_or_state_dict.state_dict()
    else:
        state_dict = module_or_state_dict

    if model_wise:
        # if model-wise, return the statistics for the entire model
        state_dict = {"model": state_dict_to_vector(state_dict)}

    for name, param in state_dict.items():
        stats[name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "min": param.data.min().item(),
            "max": param.data.max().item(),
        }

    return stats
