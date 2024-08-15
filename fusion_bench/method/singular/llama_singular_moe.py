from copy import deepcopy
from typing import List

from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaModel, LlamaForCausalLM, MistralForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .singular_moe import ExpertNotTrainedError, SingularMoELinear, SingularMoEUpscaling


class SingularMoEUpscalingForLlama(SingularMoEUpscaling):

    def merge(
        self,
        pretrained_model: LlamaForCausalLM,
        finetuned_models: List[LlamaForCausalLM],
        in_place: bool = True,
    ):
        config = self.config

        if in_place:
            model = pretrained_model
        else:
            model = deepcopy(pretrained_model)

        if isinstance(pretrained_model, (LlamaForCausalLM, MistralForCausalLM)):
            # skip the lm_head
            upscaled_model: LlamaModel = model.model
            _finetuned_models: List[LlamaModel] = [m.model for m in finetuned_models]
        else:
            raise ValueError("Nonsupported model type")

        num_layers = len(upscaled_model.layers)
        for layer_idx in tqdm(range(num_layers), "Upscaling Modules (layer)"):
            pretrained_layer: LlamaDecoderLayer = upscaled_model.layers[layer_idx]
            finetuned_layers: List[LlamaDecoderLayer] = [
                m.layers[layer_idx] for m in _finetuned_models
            ]

            if config.upscale_attn:
                self._upscale_submodules(
                    pretrained_layer.self_attn,
                    [m.self_attn for m in finetuned_layers],
                    tqdm_desc="Upscaling Linear Modules of Attntion",
                )
            if config.upscale_mlp:
                self._upscale_submodules(
                    pretrained_layer.mlp,
                    [m.mlp for m in finetuned_layers],
                    tqdm_desc="Upscaling Linear Modules of MLP",
                )
            if config.average_experts:
                self._average_experts(
                    pretrained_layer,
                    finetuned_layers,
                    "input_layernorm",
                )
                self._average_experts(
                    pretrained_layer,
                    finetuned_layers,
                    "post_attention_layernorm",
                )

        if config.upscale_lm_head:
            self._upscale_linear_layer(model, finetuned_models, "lm_head")

        return model
