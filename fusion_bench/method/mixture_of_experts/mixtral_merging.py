import logging
from typing import List, Optional, Union, cast  # noqa: F401

import torch
from omegaconf import open_dict
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
    MistralForCausalLM,
    MistralModel,
    MixtralForCausalLM,
    MixtralModel,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from fusion_bench.modelpool import BaseModelPool

from .mixtral_upcycling import (
    MixtralForCausalLMUpscalingAlgorithm,
    MixtralUpscalingAlgorithm,
    _convert_mlp,
)

log = logging.getLogger(__name__)


def _substitute_experts(
    expert_idx: int,
    expert_model: Union[LlamaModel, MistralModel],
    mixtral_model: MixtralModel,
):
    """
    Substitute the experts of the `MixtralModel` with the models from the modelpool.

    Args:
        expert_idx (int): The index of the expert to substitute.
        expert_model (Union[LlamaModel, MistralModel]): The expert model to substitute.
        mixtral_model (MixtralModel): The MixtralModel to substitute the experts in.
    """
    for input_layer, output_layer in zip(expert_model.layers, mixtral_model.layers):
        output_layer = cast(MixtralDecoderLayer, output_layer)
        input_layer = cast(Union[LlamaDecoderLayer, MistralDecoderLayer], input_layer)
        _convert_mlp(input_layer.mlp, output_layer.block_sparse_moe.experts[expert_idx])


class MixtralMoEMergingAlgorithm(MixtralUpscalingAlgorithm):
    """
    This class is responsible for merging models into a MixtralModel.
    """

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> MixtralModel:
        """
        Runs the merging process.

        Args:
            modelpool (ModelPool): The pool of models to be merged. Each model in the pool will be treated as an expert, and should be a `MistralModel` or `LlamaModel`.

        Returns:
            MixtralModel: The merged model.
        """
        with open_dict(self.config):
            self.config.num_experts = len(modelpool)

        # firstly, we upscale the models to MixtralModel
        mixtral_model = super()._run(modelpool)

        # then we substitute the experts of the MixtralModel with the models from the modelpool
        for model_idx, model_name in enumerate(modelpool.model_names):
            expert_model: MistralModel | LlamaModel = modelpool.load_model(model_name)
            _substitute_experts(model_idx, expert_model, mixtral_model)

        if self.config.get("save_checkpoint", None) is not None:
            mixtral_model.save_pretrained(self.config.save_checkpoint)
        return mixtral_model


class MixtralForCausalLMMergingAlgorithm(MixtralForCausalLMUpscalingAlgorithm):
    """
    This class is responsible for merging models into a `MixtralForCausalLM`.
    """

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> MixtralForCausalLM:
        """
        Runs the merging process. It first upscales the models to MixtralForCausalLM,
        then substitutes the experts of the MixtralForCausalLM with the models from the modelpool.

        Args:
            modelpool (ModelPool): The pool of models to be merged. Each model in the pool will be treated as an expert, and should be a `MistralForCausalLM` or `LlamaForCausalLM`.

        Returns:
            MixtralForCausalLM: The merged model.
        """
        with open_dict(self.config):
            self.config.num_experts = len(modelpool)

        # firstly, we upscale the models to MixtralForCausalLM
        mixtral_model = super()._run(modelpool)

        # then we substitute the experts of the MixtralForCausalLM with the models from the modelpool
        for model_idx, model_name in enumerate(modelpool.model_names):
            expert_model: MistralForCausalLM | LlamaForCausalLM = modelpool.load_model(
                model_name
            )
            _substitute_experts(model_idx, expert_model.model, mixtral_model.model)

        if self.config.get("save_checkpoint", None) is not None:
            mixtral_model.save_pretrained(self.config.save_checkpoint)
        return mixtral_model
