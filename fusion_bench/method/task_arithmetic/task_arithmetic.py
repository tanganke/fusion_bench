"""
This script contains the general implementation of the Task Arithmetic method.

http://arxiv.org/abs/2212.04089
"""

import logging
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, TypeVar, Union  # noqa: F401

import torch
from torch import nn

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType, TorchModelType

log = logging.getLogger(__name__)


@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: TorchModelType,
    finetuned_models: List[TorchModelType],
    scaling_factor: float,
    inplace: bool = True,
) -> TorchModelType:
    """
    Merges the task vectors from multiple fine-tuned models into a single pre-trained model.

    Args:
        pretrained_model (nn.Module): The pre-trained model to which the task vectors will be added.
        finetuned_models (List[nn.Module]): A list of fine-tuned models from which task vectors will be calculated.
        scaling_factor (float): A factor by which the task vectors will be scaled before merging.
        inplace (bool, optional): If True, the pre-trained model will be modified in place.
                                  If False, a copy of the pre-trained model will be modified. Defaults to True.

    Returns:
        nn.Module: The pre-trained model with the merged task vectors.
    """
    if not inplace:
        pretrained_model = deepcopy(pretrained_model)
    task_vector: Optional[StateDictType] = None
    # Calculate the total task vector
    for model in finetuned_models:
        if task_vector is None:
            # calculate the task vector for the first model
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )
        else:
            # calculate the task vector for the remaining models
            task_vector = state_dict_add(
                task_vector,
                state_dict_sub(
                    model.state_dict(keep_vars=True),
                    pretrained_model.state_dict(keep_vars=True),
                ),
            )
    # scale the task vector
    task_vector = state_dict_mul(task_vector, scaling_factor)
    # add the task vector to the pretrained model
    state_dict = state_dict_add(
        pretrained_model.state_dict(keep_vars=True), task_vector
    )
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model


class TaskArithmeticAlgorithm(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    """
    Task Arithmetic Algorithm for model fusion.

    This class implements the Task Arithmetic method for fusing models. It inherits from
    BaseModelFusionAlgorithm and SimpleProfilerMixin to provide the necessary functionality
    for model fusion and profiling.

    Attributes:
        scaling_factor (int): The factor by which the task vectors will be scaled before merging.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "scaling_factor": "scaling_factor"
    }

    def __init__(self, scaling_factor: int):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        """
        self.scaling_factor = scaling_factor
        super().__init__()

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]):
        """
        Runs the Task Arithmetic Algorithm to fuse models in the given model pool.

        Args:
            modelpool (Union[BaseModelPool, Dict[str, nn.Module]]): The pool of models to fuse.

        Returns:
            nn.Module: The pre-trained model with the merged task vectors.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info("Fusing models using task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        # Calculate the total task vector
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            with self.profile("merge weights"):
                if task_vector is None:
                    task_vector = state_dict_sub(
                        model.state_dict(keep_vars=True),
                        pretrained_model.state_dict(keep_vars=True),
                    )
                else:
                    task_vector = state_dict_add(
                        task_vector,
                        state_dict_sub(
                            model.state_dict(keep_vars=True),
                            pretrained_model.state_dict(keep_vars=True),
                        ),
                    )
        with self.profile("merge weights"):
            # scale the task vector
            task_vector = state_dict_mul(task_vector, self.config.scaling_factor)
            # add the task vector to the pretrained model
            state_dict = state_dict_add(
                pretrained_model.state_dict(keep_vars=True), task_vector
            )

        self.print_profile_summary()
        pretrained_model.load_state_dict(state_dict)
        return pretrained_model
