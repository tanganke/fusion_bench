import logging

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_mul

from .task_arithmetic import DareTaskArithmetic

log = logging.getLogger(__name__)


@auto_register_config
class DareSimpleAverage(BaseAlgorithm):

    def __init__(
        self,
        sparsity_ratio: float,
        only_on_linear_weights: bool,
        rescale: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sparsity_ratio = sparsity_ratio
        self.only_on_linear_weight = only_on_linear_weights
        self.rescale = rescale

    def run(self, modelpool: BaseModelPool):
        return DareTaskArithmetic(
            scaling_factor=1 / len(modelpool),
            sparsity_ratio=self.sparsity_ratio,
            only_on_linear_weights=self.only_on_linear_weight,
            rescale=self.rescale,
        ).run(modelpool)
