from typing import Literal

import torch
from torch import Tensor, nn

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method.ties_merging.ties_merging_utils import ties_merging
from fusion_bench.utils.parameters import state_dict_to_vector, vector_to_state_dict
from fusion_bench.utils.state_dict_arithmetic import state_dict_sum

from .utils import (
    module_random_drop_,
    module_sub_,
    param_random_drop_,
    trainable_state_dict,
)


class DareTiesMerging(BaseAlgorithm):
    def __init__(
        self,
        # DARE parameters
        sparsity_ratio: float,
        only_on_linear_weights: bool,
        rescale: bool,
        # Ties merging parameters
        scaling_factor: float,
        threshold: int,
        remove_keys: list[str],
        merge_func: Literal["sum", "mean", "max"],
        **kwargs,
    ):
        self.sparsity_ratio = sparsity_ratio
        self.only_on_linear_weights = only_on_linear_weights
        self.rescale = rescale
        self.scaling_factor = scaling_factor
        self.threshold = threshold
        self.remove_keys = remove_keys
        self.merge_func = merge_func
        super().__init__(**kwargs)

    @torch.no_grad()
    def _load_task_vector(
        self,
        modelpool: BaseModelPool,
        model_name: str,
        pretrained_model: nn.Module,
    ):
        finetuned_model = modelpool.load_model(model_name)
        task_vector = module_sub_(finetuned_model, pretrained_model)
        return task_vector

    def run(self, modelpool: BaseModelPool):
        assert (
            self.sparsity_ratio >= 0 and self.sparsity_ratio <= 1
        ), "Sparsity ratio must be between 0 and 1"
        pretrained_model = modelpool.load_pretrained_model()

        # load task vectors
        task_vectors = {
            model_name: self._load_task_vector(modelpool, model_name, pretrained_model)
            for model_name in modelpool.model_names
        }

        # drop and rescale task vectors
        for model_name, tv in task_vectors.items():
            if self.only_on_linear_weights:
                for module_name, module in tv.named_modules():
                    if isinstance(module, nn.Linear):
                        print(f"pruning model: `{model_name}`, layer: {module_name}.")
                        param_random_drop_(
                            module.weight, self.sparsity_ratio, rescale=self.rescale
                        )
            else:
                print(f"pruning model: `{model_name}`")
                module_random_drop_(tv, self.sparsity_ratio, rescale=self.rescale)

        ptm_check = pretrained_model.state_dict()
        flat_ptm = state_dict_to_vector(ptm_check, self.remove_keys)
        tv_flat_checks = torch.vstack(
            [
                state_dict_to_vector(check.state_dict(), self.remove_keys)
                for check in task_vectors.values()
            ]
        )
        del task_vectors

        # Perform TIES Merging
        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=self.threshold,
            merge_func=self.merge_func,
        )
        merged_check = flat_ptm + self.scaling_factor * merged_tv
        merged_state_dict = vector_to_state_dict(
            merged_check, ptm_check, remove_keys=self.remove_keys
        )

        pretrained_model.load_state_dict(merged_state_dict)
        return pretrained_model
