"""
This module contains a wrapper for switching between different models.

For example, it can be used to switch between different classification heads for a shared backbone.
"""

import logging
from typing import Dict, Optional

from torch import nn

from fusion_bench.utils.misc import first, validate_and_suggest_corrections

__all__ = ["SwitchModule", "set_active_option"]

log = logging.getLogger(__name__)


def _standardize_option_name(name: str) -> str:
    """
    Standardizes the option name by:

    - Stripping whitespace and converting to lowercase.
    - Replacing `-` with `_` if needed.
    - Replacing `/` with `_` if needed.

    Args:
        name (str): The option name to standardize.
    """
    name = name.strip().lower()
    name = name.replace("-", "_")
    name = name.replace("/", "_")
    return name


class SwitchModule(nn.Module):
    """
    A wrapper module that contains multiple sub-modules (options) and allows switching between them.

    This is useful for multi-head models or models where different parts are activated based on the task.
    """

    def __init__(self, modules: Dict[str, nn.Module]):
        """
        Args:
            modules (Dict[str, nn.Module]): A dictionary of modules to switch between.
        """
        super().__init__()
        standardized_modules = {
            _standardize_option_name(name): module for name, module in modules.items()
        }
        self._option_modules = nn.ModuleDict(standardized_modules)
        self._active_option = first(self._option_modules.keys())

    def set_active_option(self, option_name: str):
        standardized_name = _standardize_option_name(option_name)
        validate_and_suggest_corrections(standardized_name, self._option_modules.keys())
        self._active_option = standardized_name

    def forward(self, *args, **kwargs):
        active_module = self._option_modules[self._active_option]
        return active_module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            active_module = self._option_modules[self._active_option]
            if hasattr(active_module, name):
                return getattr(active_module, name)
            raise


def set_active_option(module: nn.Module, option_name: str) -> list[str]:
    """
    Utility function to set the active option for all SwitchModule instances within a given module.

    Args:
        module (nn.Module): The module to set the active option for.
        option_name (str): The name of the option to activate.

    Returns:
        list[str]: A list of names of submodules that were activated.
    """
    activated_submodules = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, SwitchModule):
            submodule.set_active_option(option_name)
            activated_submodules.append(name)
    return activated_submodules
