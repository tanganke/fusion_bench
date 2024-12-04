"""
This script contains the general implementation of the Task Arithmetic method.

http://arxiv.org/abs/2212.04089
"""

import logging
from copy import deepcopy
from typing import Dict, List, Mapping, TypeVar, Union  # noqa: F401

import torch
from torch import nn
from tqdm.autonotebook import tqdm
from collections import OrderedDict

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType
from types import SimpleNamespace

log = logging.getLogger(__name__)

class ModelScheduler:
    """
    Manage the sotrage of models, schedule the order in which models are merged in each iteration
    """
    def __init__(
        self,
        modelpool: BaseModelPool
    ):
        self.pretrained_model = modelpool.load_model("_pretrained_") # 用来算vector

        self.finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]
        self.task_vectors = [
            state_dict_sub(
                model.state_dict(keep_vars=True),
                self.pretrained_model.state_dict(keep_vars=True),
            ) for model in self.finetuned_models
        ]
        self.num_task_vectors = len(self.task_vectors)
        self.new_task_vectors = deepcopy(self.task_vectors)
        self.finetuned_models_name = [
            name for name in modelpool.model_names
        ]

    @torch.no_grad()# not sure whether to use this
    def __call__(self, model_id):
        """
        return models and relevant data in each step
        """
        # TODO: use a mixing matrix to determine which models to use in step idx

        task_vectors = [
            deepcopy(self.task_vectors[(model_id+1)%self.num_task_vectors]),
            deepcopy(self.task_vectors[model_id]),
            deepcopy(self.task_vectors[(model_id-1)%self.num_task_vectors])
        ]

        finetuned_models_names = [self.finetuned_models_name[(model_id+i)%self.num_task_vectors] for i in range(-1, 2, 1)]

        return task_vectors, finetuned_models_names
    
    def store_model(self, new_task_vectors, model_id):
        """
        store new finetuned model after every turn of adamerging
        """
        self.new_task_vectors[model_id]=deepcopy(new_task_vectors)

    def update_models(self):
        self.task_vectors = deepcopy(self.new_task_vectors)

    def get_final_models(self, pretrained_model):
        # need a check
        final_models_state_dict = [{'name': name, 'model': model} for name, model in zip(self.finetuned_models_name, self.task_vectors)]
        num_finetuned_models = len(self.task_vectors)
    
        average_model = OrderedDict()
        
        for k in self.task_vectors[0]:
            for i in self.task_vectors:
                if k not in average_model:
                    average_model[k] = i[k]
                else:
                    average_model[k] = average_model[k] + i[k]
            average_model[k] = average_model[k] / num_finetuned_models

        final_models_state_dict += [{'name': 'average model', 'model': average_model}]
        
        final_models = []
        for Model in final_models_state_dict:
            Model['model'] = state_dict_add(
                self.pretrained_model.state_dict(keep_vars=True), Model['model']
            )
            pretrained_model.load_state_dict(Model['model'])
            final_models.append({'name': Model['name'], 'model': deepcopy(pretrained_model)})

        return final_models
    

@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: nn.Module,
    finetuned_models: List[nn.Module],
    scaling_factor: float,
    inplace: bool = True,
) -> nn.Module:
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
    task_vector: StateDictType = None
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

class TaskArithmetic_Gossip_Algorithm(
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

    def __init__(self, scaling_factor: int, **kwargs):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        """
        self.configs = SimpleNamespace(**kwargs)
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

        self.modelpool = modelpool
        self.num_finetuned_models = len(modelpool.model_names)

        log.info("Fusing models using task arithmetic.")

        pretrained_model = modelpool.load_model("_pretrained_")  # 作为容器，不使用实际参数
        
        model_scheduler = ModelScheduler(self.modelpool)

        for step_idx in tqdm(
            range(self.configs.gossip_max_steps),
            "Gossip merging",
            dynamic_ncols=True
        ):
            log.info(f'Gossip merging step:, {step_idx}')
            for model_id in tqdm(
                range(self.num_finetuned_models),
                "local admerging",
                dynamic_ncols=True
            ):
                task_vector = None
                # Calculate the total task vector
                module_state_dict, name = model_scheduler(model_id)
                for model_state_dict in module_state_dict:

                    if task_vector is None:
                        task_vector = deepcopy(model_state_dict)
                    else:
                        task_vector = state_dict_add(
                            task_vector,
                            deepcopy(model_state_dict)
                        )

                # scale the task vector
                task_vector = state_dict_mul(task_vector, self.config.scaling_factor)
                # add the task vector to the pretrained model
                # state_dict = state_dict_add(
                #     pretrained_model.state_dict(keep_vars=True), task_vector
                # )
                model_scheduler.store_model(task_vector, model_id)
            
            model_scheduler.update_models()
            do_evaluation = False # whether to do evaluation after each Gossip step
            if isinstance(self.configs.accuracy_test_interval, list):
                if (step_idx+1) in self.configs.accuracy_test_interval:
                    do_evaluation = True
            elif isinstance(self.configs.accuracy_test_interval, int):
                if self.configs.accuracy_test_interval != 0 and ((step_idx+1) % self.configs.accuracy_test_interval == 0):
                    do_evaluation = True
            if do_evaluation:
                self._program.evaluate_merged_model(self._program.taskpool, model_scheduler.get_final_models(pretrained_model))
    
        return model_scheduler.get_final_models(pretrained_model)
