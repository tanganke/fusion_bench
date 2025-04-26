import logging
from typing import Dict, List, Literal, Mapping, Union  # noqa: F401

import torch
from torch import Tensor, nn

from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType
from tqdm.autonotebook import tqdm

from ..ties_merging.ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict
from copy import deepcopy
from types import SimpleNamespace

log = logging.getLogger(__name__)

class ModelScheduler:
    """
    Manage the sotrage of models, schedule the order in which models are merged in each iteration
    """
    def __init__(
        self,
        modelpool: BaseModelPool,
        remove_keys: List[str],
    ):
        self.pretrained_model = modelpool.load_model("_pretrained_") # 用来算vector
        self.finetuned_models_name = [
            name for name in modelpool.model_names
        ]
        self.remove_keys = remove_keys

        # Load the state dicts of the models
        ft_checks: List[StateDictType] = [
            modelpool.load_model(model_name).state_dict(keep_vars=True)
            for model_name in modelpool.model_names
        ]
        self.ptm_check: StateDictType = self.pretrained_model.state_dict(keep_vars=True)

        # Compute the task vectors
        flat_ft: Tensor = torch.vstack(
            [state_dict_to_vector(check, remove_keys) for check in ft_checks]
        )
        self.flat_ptm: Tensor = state_dict_to_vector(self.ptm_check, remove_keys)
        self.tv_flat_checks = flat_ft - self.flat_ptm

        self.new_tv_flat_checks = deepcopy(self.tv_flat_checks)
        self.num_task_vectors = len(modelpool.model_names)

    @torch.no_grad()# not sure whether to use this
    def __call__(self, model_id):
        """
        return models and relevant data in each step
        """
        # TODO: use a mixing matrix to determine which models to use in step idx

        task_vectors = torch.vstack([
            deepcopy(self.tv_flat_checks[(model_id+1)%self.num_task_vectors]),
            deepcopy(self.tv_flat_checks[model_id]),
            deepcopy(self.tv_flat_checks[(model_id-1)%self.num_task_vectors])
        ])

        task_vectors_name = [self.finetuned_models_name[(model_id+i)%self.num_task_vectors] for i in range(-1, 2, 1)]
        return task_vectors, task_vectors_name
    
    def store_model(self, new_task_vectors, model_id):
        """
        store new finetuned model after every turn of adamerging
        """
        self.new_tv_flat_checks[model_id]=deepcopy(new_task_vectors)

    def update_models(self):
        self.tv_flat_checks = deepcopy(self.new_tv_flat_checks)

    def get_final_models(self, pretrained_model):
        # need a check
        final_models_state_dict = [{'name': name, 'model': model} for name, model in zip(self.finetuned_models_name, self.tv_flat_checks)]
        num_finetuned_models = len(self.tv_flat_checks)
    
        average_model = deepcopy(self.tv_flat_checks[0])
        
        for i in range(1, num_finetuned_models):
            average_model += self.tv_flat_checks[i]
        average_model /= num_finetuned_models

        final_models_state_dict += [{'name': 'average model', 'model': average_model}]
        
        final_models = []
        for Model in final_models_state_dict:
            Model['model'] += self.flat_ptm
            Model['model'] = vector_to_state_dict(
                Model['model'], self.ptm_check, remove_keys=self.remove_keys
            )
            pretrained_model.load_state_dict(Model['model'])
            final_models.append({'name': Model['name'], 'model': deepcopy(pretrained_model)})

        return final_models

def task_arithmetic(tv_flat_checks):
    ans = None
    for i in tv_flat_checks:
        if ans == None:
            ans = i
        else:
            ans += i
    return ans

class TiesMerging_Gossip_Algorithm(BaseAlgorithm):
    """
    TiesMergingAlgorithm is a class for fusing multiple models using the TIES merging technique.

    Attributes:
        scaling_factor (float): The scaling factor to apply to the merged task vector.
        threshold (float): The threshold for resetting values in the task vector.
        remove_keys (List[str]): List of keys to remove from the state dictionary.
        merge_func (Literal["sum", "mean", "max"]): The merge function to use for disjoint merging.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "scaling_factor": "scaling_factor",
        "threshold": "threshold",
        "remove_keys": "remove_keys",
        "merge_func": "merge_func",
    }

    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: List[str],
        merge_func: Literal["sum", "mean", "max"],
        **kwargs,
    ):
        """
        Initialize the TiesMergingAlgorithm with the given parameters.

        Args:
            scaling_factor (float): The scaling factor to apply to the merged task vector.
            threshold (float): The threshold for resetting values in the task vector.
            remove_keys (List[str]): List of keys to remove from the state dictionary.
            merge_func (Literal["sum", "mean", "max"]): The merge function to use for disjoint merging.
            **kwargs: Additional keyword arguments for the base class.
        """
        self.scaling_factor = scaling_factor
        self.threshold = threshold
        self.remove_keys = remove_keys
        self.merge_func = merge_func
        self.configs = SimpleNamespace(**kwargs)
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs):
        """
        Run the TIES merging algorithm to fuse models in the model pool.

        Args:
            modelpool (BaseModelPool | Dict[str, nn.Module]): The model pool containing the models to fuse.

        Returns:
            nn.Module: The fused model.
        """
        log.info("Fusing models using ties merging.")
        modelpool = to_modelpool(modelpool)
        remove_keys = self.config.get("remove_keys", [])
        merge_func = self.config.get("merge_func", "sum")
        ties_merging_steps = self.configs.ties_merging_steps # it will become task arithmetic if equal to gossip_max_steps
        task_using_task_arithmetic = self.configs.task_using_task_arithmetic # not all task apply to ties-merging when only merge three models
        print(type(task_using_task_arithmetic))
        scaling_factor = self.scaling_factor
        threshold = self.threshold

        model_scheduler = ModelScheduler(modelpool, remove_keys)
        self.num_finetuned_models = len(modelpool.model_names)
        client_name = modelpool.model_names

        # Load the pretrained model
        pretrained_model = modelpool.load_model("_pretrained_")

        for step_idx in tqdm(
            range(self.configs.gossip_max_steps),
            "Gossip merging",
            dynamic_ncols=True
        ):
            print('ties_merging_step:', int(ties_merging_steps))
            if step_idx > int(ties_merging_steps):
                local_merge_method = 'ties_merging'
            else:
                local_merge_method = 'task_arithmetic'
            log.info(f'Gossip merging step:, {step_idx}')
            for model_id in tqdm(
                range(self.num_finetuned_models),
                "local admerging",
                dynamic_ncols=True
            ):
                tv_flat_checks, merging_clients_name = model_scheduler(model_id)
                if client_name[model_id] in task_using_task_arithmetic:
                    local_merge_method = 'task_arithmetic'
                log.info(f'merging clients: {merging_clients_name}')
                log.info(f'using {local_merge_method}')
                if local_merge_method == 'ties_merging':
                    # Perform TIES Merging
                    merged_tv = ties_merging(
                        tv_flat_checks,
                        reset_thresh=threshold,
                        merge_func=merge_func,
                    )
                    merged_check = scaling_factor * merged_tv
                    model_scheduler.store_model(merged_check, model_id)
                elif local_merge_method == 'task_arithmetic':
                    # Perform Task Arithmetic
                    merged_tv = task_arithmetic(tv_flat_checks)
                    merged_check = scaling_factor * merged_tv
                    model_scheduler.store_model(merged_check, model_id)
                else:
                    raise ValueError(f"Unknown merge method: {local_merge_method}")

            model_scheduler.update_models()
            do_evaluation = False # whether to do evaluation after each Gossip step
            if isinstance(self.configs.accuracy_test_interval, list):
                if (step_idx+1) in self.configs.accuracy_test_interval:
                    do_evaluation = True
            elif isinstance(self.configs.accuracy_test_interval, int):
                if self.configs.accuracy_test_interval != 0 and ((step_idx+1) % self.configs.accuracy_test_interval == 0):
                    do_evaluation = True
            if do_evaluation:
                self._program.evaluate_merged_model(self._program.taskpool, model_scheduler.get_final_models(pretrained_model))   ## There may still have some bug
    
        return model_scheduler.get_final_models(pretrained_model)