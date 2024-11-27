import logging
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Union

import torch
from torch import nn

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_avg,
    state_dict_div,
    state_dict_mul,
)
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


def simple_average(
    modules: List[Union[nn.Module, StateDictType]],
    base_module: Optional[nn.Module] = None,
):
    R"""
    Averages the parameters of a list of PyTorch modules or state dictionaries.

    This function takes a list of PyTorch modules or state dictionaries and returns a new module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Args:
        modules (List[Union[nn.Module, StateDictType]]): A list of PyTorch modules or state dictionaries.
        base_module (Optional[nn.Module]): A base module to use for the new module. If provided, the averaged parameters will be loaded into this module. If not provided, a new module will be created by copying the first module in the list.

    Returns:
        module_or_state_dict (Union[nn.Module, StateDictType]): A new PyTorch module with the averaged parameters, or a new state dictionary with the averaged parameters.

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
        if base_module is None:
            new_module = deepcopy(modules[0])
        else:
            new_module = base_module
        state_dict = state_dict_avg([module.state_dict() for module in modules])
        new_module.load_state_dict(state_dict)
        return new_module
    elif isinstance(modules[0], Mapping):
        return state_dict_avg(modules)


class SimpleAverageAlgorithm(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]):
        """
        Fuse the models in the given model pool using simple averaging.

        This method iterates over the names of the models in the model pool, loads each model, and appends it to a list.
        It then returns the simple average of the models in the list.

        Args:
            modelpool: The pool of models to fuse.

        Returns:
            The fused model obtained by simple averaging.
        """
        if isinstance(modelpool, dict):
            modelpool = BaseModelPool(modelpool)

        log.info(
            f"Fusing models using simple average on {len(modelpool.model_names)} models."
            f"models: {modelpool.model_names}"
        )
        sd: Optional[StateDictType] = None
        forward_model = None
        merged_model_names = []

        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
                merged_model_names.append(model_name)
                print(f"load model of type: {type(model).__name__}")
            with self.profile("merge weights"):
                if sd is None:
                    # Initialize the state dictionary with the first model's state dictionary
                    sd = model.state_dict(keep_vars=True)
                    forward_model = model
                else:
                    # Add the current model's state dictionary to the accumulated state dictionary
                    sd = state_dict_add(sd, model.state_dict(keep_vars=True))
        with self.profile("merge weights"):
            # Divide the accumulated state dictionary by the number of models to get the average
            sd = state_dict_div(sd, len(modelpool.model_names))

        forward_model.load_state_dict(sd)
        # print profile report and log the merged models
        self.print_profile_summary()
        log.info(f"merged {len(merged_model_names)} models:")
        for model_name in merged_model_names:
            log.info(f"  - {model_name}")
        return forward_model
