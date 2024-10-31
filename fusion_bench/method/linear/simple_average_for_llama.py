from typing import Optional

from typing_extensions import override

from fusion_bench import timeit_context
from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.method.simple_average import SimpleAverageAlgorithm
from fusion_bench.modelpool import CausalLMBackbonePool, CausalLMPool


class SimpleAverageForLlama(BaseAlgorithm):
    R"""
    A simple averaging algorithm for LLama models. If `merge_backbone` is set to `True`, the backbone of the model will be averaged and the rest of the model will be loaded from the pre-trained model.

    Examples:

        The following example demonstrates how to use the `SimpleAverageForLlama` algorithm to merge Mistral models.

        ```bash
        fusion_bench \
            method=linear/simple_average_for_llama \
            method.model_save_path=outputs/simle_mixtral_exp_v4/simple_average \
            modelpool=CausalLMPool/simle_mixtral_exp_v4.yaml
        ```
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "merge_backbone": "merge_backbone",
    }

    def __init__(self, merge_backbone: bool, model_save_path: Optional[str] = None):
        super().__init__()
        self.merge_backbone = merge_backbone
        self.model_save_path = model_save_path

    @override
    def run(self, modelpool: CausalLMPool):
        if self.model_save_path:
            tokenizer = modelpool.load_tokenizer()

        if self.merge_backbone:
            assert modelpool.has_pretrained
            backbone_modelpool = CausalLMBackbonePool(**modelpool.config)
            model = modelpool.load_model("_pretrained_")
            backbone_model = SimpleAverageAlgorithm().run(backbone_modelpool)
            model.model.layers = backbone_model
        else:
            model = SimpleAverageAlgorithm().run()

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
        return model
