import logging
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Union

import torch
from torch import nn
from tqdm.autonotebook import tqdm

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_avg,
    state_dict_mul,
    state_dict_div,
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
        # simple average don't need a pretrained model
        # self.pretrained_model = modelpool.load_model("_pretrained_")# 

        self.finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]
        self.num_finetuned_models = len(self.finetuned_models)
        self.new_finetuned_models = deepcopy(self.finetuned_models)
        self.finetuned_models_name = [
            name for name in modelpool.model_names
        ]

    @torch.no_grad()# not sure whether to use this
    def __call__(self, model_id):
        """
        return models and relevant data in each step
        """
        # TODO: use a mixing matrix to determine which models to use in step idx

        finetuned_models = [
            deepcopy(self.finetuned_models[(model_id+1)%self.num_finetuned_models]),
            deepcopy(self.finetuned_models[model_id]),
            deepcopy(self.finetuned_models[(model_id-1)%self.num_finetuned_models])
        ]

        finetuned_models_names = [self.finetuned_models_name[(model_id+i)%self.num_finetuned_models] for i in range(-1, 2, 1)]

        return finetuned_models, finetuned_models_names
    
    def store_model(self, new_finetuned_model_dict, model_id):
        """
        store new finetuned model after every turn of adamerging
        """
        self.new_finetuned_models[model_id].load_state_dict(new_finetuned_model_dict)

    def update_models(self):
        self.finetuned_models = deepcopy(self.new_finetuned_models)

    def get_final_models(self):
        # need a check
        final_models = [{'name': name, 'model': model} for name, model in zip(self.finetuned_models_name, self.finetuned_models)]
        num_finetuned_models = len(self.finetuned_models)
    
        average_model = deepcopy(self.finetuned_models[0])
        state_dict = average_model.state_dict(keep_vars=True)
        for name, _ in self.finetuned_models[0].named_parameters():
            state_dict[name].data.zero_()
        for model in self.finetuned_models:
            for name, param in model.named_parameters():
                state_dict[name] = state_dict[name] + 1/num_finetuned_models * param
            
        average_model.load_state_dict(state_dict)
        final_models += [{'name': 'average model', 'model': average_model}]
        
        return final_models
    
    def move_to(self, device):
        # self.pretrained_model.to(device=device)
        for model in self.finetuned_models:
            model.to(device=device)

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


class SimpleAverage_Gossip_Algorithm(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    
    def __init__(
        self,
        **kwargs
    ):
        self.configs = SimpleNamespace(**kwargs)
        super().__init__(**kwargs)
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

        self.modelpool = modelpool
        self.num_finetuned_models = len(modelpool.model_names)

        log.info(
            f"Fusing models using simple average on {len(modelpool.model_names)} models."
            f"models: {modelpool.model_names}"
        )

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
                module, name = model_scheduler(model_id)
                sd: Optional[StateDictType] = None
                # merged_model_names = []
                log.info(f'merge model: {name}')
                for model in module:
                    # with self.profile("load model"):
                    # model = modelpool.load_model(model_name)
                    # merged_model_names.append(model_name)
                    # print(f"load model of type: {type(model).__name__}")
                    # with self.profile("merge weights"):
                    if sd is None:
                        # Initialize the state dictionary with the first model's state dictionary
                        sd = model.state_dict(keep_vars=True)
                    else:
                        # Add the current model's state dictionary to the accumulated state dictionary
                        sd = state_dict_add(sd, model.state_dict(keep_vars=True))
                # with self.profile("merge weights"):
                    # Divide the accumulated state dictionary by the number of models to get the average
                sd = state_dict_div(sd, len(module))
                model_scheduler.store_model(sd, model_id)

            model_scheduler.update_models()
            do_evaluation = False # whether to do evaluation after each Gossip step
            if isinstance(self.configs.accuracy_test_interval, list):
                if (step_idx+1) in self.configs.accuracy_test_interval:
                    do_evaluation = True
            elif isinstance(self.configs.accuracy_test_interval, int):
                if ((step_idx+1) % self.configs.accuracy_test_interval == 0):
                    do_evaluation = True
            if do_evaluation:
                self._program.evaluate_merged_model(self._program.taskpool, model_scheduler.get_final_models())
                model_scheduler.move_to('cpu')
        # print profile report and log the merged models
        # self.print_profile_summary()
        # log.info(f"merged {len(merged_model_names)} models:")
        # for model_name in merged_model_names:
        #     log.info(f"  - {model_name}")
        return model_scheduler.get_final_models()
