import logging
from copy import deepcopy
from typing import List, Mapping, Union

import torch
from torch import Tensor, nn

from ..utils.state_dict_arithmetic import state_dict_avg
from ..utils.type import _StateDict
from .base_algorithm import ModelFusionAlgorithm

log = logging.getLogger(__name__)


def simple_average(modules: List[Union[nn.Module, _StateDict]]):
    """
    Averages the parameters of a list of PyTorch modules or state dictionaries.

    This function takes a list of PyTorch modules or state dictionaries and returns a new module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Args:
        modules (List[Union[nn.Module, _StateDict]]): A list of PyTorch modules or state dictionaries.

    Returns:
        module_or_state_dict (Union[nn.Module, _StateDict]): A new PyTorch module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Examples:
        >>> import torch.nn as nn
        >>> model1 = nn.Linear(10, 10)
        >>> model2 = nn.Linear(10, 10)
        >>> averaged_model = simple_averageing([model1, model2])

        >>> state_dict1 = model1.state_dict()
        >>> state_dict2 = model2.state_dict()
        >>> averaged_state_dict = simple_averageing([state_dict1, state_dict2])
    """
    if isinstance(modules[0], nn.Module):
        new_module = deepcopy(modules[0])
        state_dict = state_dict_avg([module.state_dict() for module in modules])
        new_module.load_state_dict(state_dict)
        return new_module
    elif isinstance(modules[0], Mapping):
        return state_dict_avg(modules)


class SimpleAverageAlgorithm(ModelFusionAlgorithm):
    def fuse(self, modelpool):
        log.info("Fusing models using simple average.")
        log.info("Loading models.")
        models = []
        for model_name in modelpool.model_names:
            model = modelpool.load_model(model_name)
            models.append(model)

        log.info("Fusing models.")
        return simple_average(models)
