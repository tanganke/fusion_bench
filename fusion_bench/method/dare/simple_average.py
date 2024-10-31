import logging

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_mul

from .utils import module_random_drop_, trainable_state_dict

log = logging.getLogger(__name__)


class DareSimpleAverage(BaseAlgorithm):

    def __init__(
        self,
        sparsity_ratio: float,
        only_on_linear_weights: bool,
        rescale: bool = True,
        **kwargs,
    ):
        self.sparsity_ratio = sparsity_ratio
        self.only_on_linear_weight = only_on_linear_weights
        self.rescale = rescale
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        if modelpool.has_pretrained:
            log.warning("Pretrained model provided but not used")

        sum_state_dict = None
        for model in modelpool.models():
            if sum_state_dict is None:
                sum_state_dict = trainable_state_dict(model)
                sum_state_dict = module_random_drop_(
                    sum_state_dict, self.sparsity_ratio, rescale=self.rescale
                )
            else:
                state_dict = trainable_state_dict(model)
                state_dict = module_random_drop_(
                    model, self.sparsity_ratio, rescale=self.rescale
                )
                sum_state_dict = state_dict_add(sum_state_dict, state_dict)
        state_dict = state_dict_mul(sum_state_dict, len(modelpool))

        model.load_state_dict(state_dict, strict=False)
        return model
