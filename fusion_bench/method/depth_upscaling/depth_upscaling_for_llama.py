import os
from typing import Optional

from typing_extensions import override

from fusion_bench.modelpool.causal_lm.causal_lm import CausalLM, CausalLMPool
from fusion_bench.utils import timeit_context

from .depth_upscaling import DepthUpscalingAlgorithm


class DepthUpscalingForLlama(DepthUpscalingAlgorithm):

    def __init__(self, layer_indices: list, model_save_path: Optional[str], **kwargs):
        if isinstance(model_save_path, str):
            model_save_path = os.path.expanduser(model_save_path)
        self.model_save_path = model_save_path
        super().__init__(layer_indices, **kwargs)

    @override
    def run(self, modelpool: CausalLMPool):
        if self.model_save_path is not None:
            tokenizer = modelpool.load_tokenizer()

        model: CausalLM = modelpool.load_pretrained_or_first_model()
        model.model.layers = super().run(model.model.layers)
        model.config.num_hidden_layers = len(model.model.layers)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
        return model
