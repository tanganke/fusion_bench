from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from fusion_bench.utils.dict import dict_merge
from fusion_bench.utils.type import StateDictType, TorchModelType


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


def get_target_state_dict(
    module: nn.Module,
    target_modules: str | Iterable[str] | None = None,
    prefix: str = "",
    keep_vars: bool = False,
) -> StateDictType:
    """
    This function retrieves the state dictionary of specified target submodules within a given module
    of a PyTorch model or merged state dictionary from multiple submodules.

    For example, if a model has submodules named "layer1", "layer2", and "layer3", and you want to get the state dictionary of "layer1" and "layer3",
    you can call this function with `target_modules` set to `["layer1", "layer3"]`.
    The function will return a state dictionary that includes only the parameters and buffers from those specified submodules.

    Args:
        module (nn.Module): The PyTorch module containing the target submodules.
        target_modules (str | Iterable[str]): A single target module name or an iterable of target module names.
            If None, the entire module's state dictionary is returned if no special attribute is set (look up the `_fusion_bench_target_modules` attribute).
        keep_vars (bool): If True, keeps the variables in the state dictionary. Default is False.

    Returns:
        StateDictType: The state dictionary of the specified target submodules, merged if multiple are provided.
    """
    if target_modules is None:
        if (
            hasattr(module, "_fusion_bench_target_modules")
            and module._fusion_bench_target_modules is not None
        ):
            return get_target_state_dict(
                module,
                target_modules=module._fusion_bench_target_modules,
                prefix=prefix,
                keep_vars=keep_vars,
            )
        else:
            return module.state_dict(prefix=prefix, keep_vars=keep_vars)

    if isinstance(target_modules, str):
        target_modules = [target_modules]

    state_dicts = []
    for target_module in target_modules:
        submodule_prefix = (
            f"{prefix}{target_module}." if prefix else f"{target_module}."
        )
        submodule = module.get_submodule(target_module)
        state_dict = submodule.state_dict(prefix=submodule_prefix, keep_vars=keep_vars)
        state_dicts.append(state_dict)

    merged_state_dict = dict_merge(state_dicts, disjoint=True)
    return merged_state_dict


def validate_target_modules_equal(modules: Iterable[nn.Module]) -> None:
    """
    Validates that the `_fusion_bench_target_modules` attribute is the same across all provided modules.

    Args:
        modules (Iterable[nn.Module]): An iterable of PyTorch modules to validate.

    Raises:
        ValueError: If the `_fusion_bench_target_modules` attribute differs among the modules.
    """
    model_iter = iter(modules)
    first_module = next(model_iter)

    if hasattr(first_module, "_fusion_bench_target_modules"):
        target_modules = first_module._fusion_bench_target_modules
    else:
        # if the module does not have the attribute, set to None
        target_modules = None

    for module in model_iter:
        if target_modules is None:
            if (
                hasattr(module, "_fusion_bench_target_modules")
                and module._fusion_bench_target_modules != target_modules
            ):
                raise ValueError(
                    "_fusion_bench_target_modules attribute differs among the provided modules."
                )
        else:
            if (
                not hasattr(module, "_fusion_bench_target_modules")
                or module._fusion_bench_target_modules != target_modules
            ):
                raise ValueError(
                    "_fusion_bench_target_modules attribute differs among the provided modules."
                )


def load_state_dict_into_target_modules(
    module: TorchModelType,
    state_dict: StateDictType,
    target_modules: str | Iterable[str] | None = None,
    strict: bool = True,
    assign: bool = False,
):
    """
    Load a state dictionary into specified target submodules within a given module of a PyTorch model.

    This function allows you to load parameters and buffers from a state dictionary into specific submodules
    of a PyTorch model. If the `target_modules` argument is provided, only the specified submodules will be updated
    with the corresponding entries from the state dictionary.

    Args:
        module (nn.Module): The PyTorch module containing the target submodules.
        state_dict (StateDictType): The state dictionary containing parameters and buffers to load.
        target_modules (str | Iterable[str]): A single target module name or an iterable of target module names.
            If None, the entire module's state dictionary is updated if no special attribute is set
            (look up the `_fusion_bench_target_modules` attribute).
        strict (bool): Whether to strictly enforce that the keys in `state_dict` match the keys returned by
            the module's `state_dict()` function. Default is True.
    """
    if target_modules is None:
        if (
            hasattr(module, "_fusion_bench_target_modules")
            and module._fusion_bench_target_modules is not None
        ):
            return load_state_dict_into_target_modules(
                module,
                state_dict,
                target_modules=module._fusion_bench_target_modules,
                strict=strict,
                assign=assign,
            )
        else:
            return module.load_state_dict(state_dict, strict=strict, assign=assign)

    if isinstance(target_modules, str):
        target_modules = [target_modules]

    assert (
        len(target_modules) > 0
    ), "target_modules should contain at least one module name."
    results: list[_IncompatibleKeys] = []
    for target_module in target_modules:
        submodule_prefix = f"{target_module}."
        submodule_prefix_len = len(submodule_prefix)
        submodule = module.get_submodule(target_module)

        # Extract the relevant portion of the state dictionary for the submodule
        submodule_state_dict = {
            key[submodule_prefix_len:]: value for key, value in state_dict.items()
        }

        # Load the extracted state dictionary into the submodule
        result = submodule.load_state_dict(
            submodule_state_dict, strict=strict, assign=assign
        )
        results.append(result)

    # Merge results from all submodules
    merged_result = _IncompatibleKeys(
        missing_keys=[key for res in results for key in res.missing_keys],
        unexpected_keys=[key for res in results for key in res.unexpected_keys],
    )
    return merged_result
