import torch
from torch import Tensor, nn

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_sum

from .utils import (
    module_sub_,
    module_random_drop_,
    param_random_drop_,
)


class DareTaskArithmetic(BaseAlgorithm):
    """
    Implementation of Task Arithmetic w/ DARE.

    - Yu et al. Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch. 2023. http://arxiv.org/abs/2311.03099
    """

    def __init__(
        self,
        scaling_factor: float,
        sparsity_ratio: float,
        only_on_linear_weights: bool,
        **kwargs,
    ):
        self.scaling_factor = scaling_factor
        self.sparsity_ratio = sparsity_ratio
        self.only_on_linear_weights = only_on_linear_weights
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        assert (
            self.sparsity_ratio >= 0 and self.sparsity_ratio <= 1
        ), "Sparsity ratio must be between 0 and 1"
        pretrained_model = modelpool.load_pretrained_model()
        finetuned_models = {
            model_name: modelpool.load_model(model_name)
            for model_name in modelpool.model_names
        }
        task_vectors = {
            model_name: module_sub_(finetuned_models, pretrained_model)
            for model_name in finetuned_models
        }
        del finetuned_models

        # drop and rescale task vectors
        for tv in task_vectors.values():
            if self.only_on_linear_weights:
                for module in tv.modules():
                    if isinstance(module, nn.Linear):
                        param_random_drop_(
                            module.weight, self.sparsity_ratio, rescale=True
                        )
            else:
                module_random_drop_(tv, self.sparsity_ratio, rescale=True)

        # merge task vectors
        task_vector_sum = state_dict_sum(task_vectors.values())

        # scale the task vector and add it to the pretrained model
        for name, delta in task_vector_sum.items():
            delta = delta * self.scaling_factor
            pretrained_model.get_parameter(name).data.add_(delta)

        return pretrained_model
