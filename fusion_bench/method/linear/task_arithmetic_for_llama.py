import logging
from typing import Dict, List, Mapping, Optional, TypeVar, Union  # noqa: F401

from typing_extensions import override

from fusion_bench import timeit_context
from fusion_bench.method import TaskArithmeticAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMBackbonePool, CausalLMPool

log = logging.getLogger(__name__)


class TaskArithmeticForLlama(TaskArithmeticAlgorithm, SimpleProfilerMixin):
    R"""
    Examples:

    fusion_bench \
        method=linear/task_arithmetic_for_llama \
            method.scaling_factor=0.3 \
        method.model_save_path=outputs/simle_mixtral_exp_v4/task_arithmetic_0.3 \
        modelpool=CausalLMPool/simle_mixtral_exp_v4.yaml
    """

    _config_mapping = TaskArithmeticAlgorithm._config_mapping | {
        "merge_backbone": "merge_backbone",
    }

    def __init__(
        self,
        scaling_factor: float,
        merge_backbone: bool,
        model_save_path: Optional[str] = None,
    ):
        self.merge_backbone = merge_backbone
        self.model_save_path = model_save_path
        super().__init__(scaling_factor=scaling_factor)

    @override
    def run(self, modelpool: CausalLMPool):
        if self.model_save_path:
            tokenizer = modelpool.load_tokenizer()

        if self.merge_backbone:
            assert modelpool.has_pretrained
            backbone_modelpool = CausalLMBackbonePool(**modelpool.config)
            model = modelpool.load_model("_pretrained_")
            backbone_model = super().run(backbone_modelpool)
            model.model.layers = backbone_model
        else:
            model = super().run(modelpool)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
        return model
