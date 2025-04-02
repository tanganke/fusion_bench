import torch
from torch import Tensor, nn

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_sum

from .utils import (
    module_random_drop_,
    module_sub_,
    param_random_drop_,
    trainable_state_dict,
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
        rescale: bool = True,
        **kwargs,
    ):
        self.scaling_factor = scaling_factor
        self.sparsity_ratio = sparsity_ratio
        self.only_on_linear_weights = only_on_linear_weights
        self.rescale = rescale
        super().__init__(**kwargs)

    def _load_task_vector(
        self,
        modelpool: BaseModelPool,
        model_name: str,
        pretrained_model: nn.Module,
    ):
        finetuned_model = modelpool.load_model(model_name)
        task_vector = module_sub_(finetuned_model, pretrained_model)
        return task_vector

    @torch.no_grad()
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

        # merge task vectors
        task_vector_sum = state_dict_sum(
            [trainable_state_dict(tv) for tv in task_vectors.values()]
        )

        # scale the task vector and add it to the pretrained model
        for name, delta in task_vector_sum.items():
            delta = delta * self.scaling_factor
            pretrained_model.get_parameter(name).data.add_(delta)

        return pretrained_model
