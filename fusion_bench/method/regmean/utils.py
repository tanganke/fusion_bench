import re
from typing import Dict, List

from torch import nn


def get_param_names_to_merge(
    input_param_names: List[str], exclude_param_names_regex: list
) -> List[str]:
    """
    get the names of parameters that need to be merged

    Args:
        input_param_names: list, names of input parameters
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded

    Returns:
        list: names of parameters that need to be merged
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any(
            [
                re.match(exclude_pattern, param_name)
                for exclude_pattern in exclude_param_names_regex
            ]
        )
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(
    model: nn.Module, include_module_types: list
) -> Dict[str, nn.Module]:
    """
    get the model modules that need to be merged, whose type is in include_module_types

    Args:
        model: nn.Module, input model
        include_module_types: list, module types that want to include

    Returns:
        Dict[str, nn.Module]: a dictionary of modules to merge
    """
    modules_to_merge: Dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any(
            [
                isinstance(module, include_module_type)
                for include_module_type in include_module_types
            ]
        )
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge
