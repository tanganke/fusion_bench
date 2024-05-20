import logging
from copy import deepcopy
from typing import List, Mapping, TypeVar, Union

import torch
from torch import Tensor, nn

from ..modelpool import ModelPool, to_modelpool
from ..utils.state_dict_arithmetic import state_dict_add, state_dict_mul, state_dict_sub
from ..utils.type import _StateDict
from .base_algorithm import ModelFusionAlgorithm

Module = TypeVar("Module")

log = logging.getLogger(__name__)


@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: Module,
    finetuned_models: List[Module],
    scaling_factor: float,
) -> Module:
    """
    Attention: This function changes the pretrained_model in place.
    """
    task_vector = None
    # Calculate the total task vector
    for model in finetuned_models:
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
    # scale the task vector
    task_vector = state_dict_mul(task_vector, scaling_factor)
    # add the task vector to the pretrained model
    state_dict = state_dict_add(
        pretrained_model.state_dict(keep_vars=True), task_vector
    )
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model


class TaskArithmeticAlgorithm(ModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        modelpool = to_modelpool(modelpool)
        log.info("Fusing models using task arithmetic.")
        task_vector = None
        pretrained_model = modelpool.load_model("_pretrained_")

        # Calculate the total task vector
        for model_name in modelpool.model_names:
            model = modelpool.load_model(model_name)
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
        # scale the task vector
        task_vector = state_dict_mul(task_vector, self.config.scaling_factor)
        # add the task vector to the pretrained model
        state_dict = state_dict_add(
            pretrained_model.state_dict(keep_vars=True), task_vector
        )
        pretrained_model.load_state_dict(state_dict)

        return pretrained_model
