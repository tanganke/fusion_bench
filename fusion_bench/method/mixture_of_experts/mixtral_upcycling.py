import logging
from typing import Optional

import torch
from tqdm.autonotebook import tqdm
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    MistralConfig,
    MistralForCausalLM,
    MistralModel,
    MixtralConfig,
    MixtralForCausalLM,
    MixtralModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralMLP
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralDecoderLayer,
)
from transformers.utils import ContextManagers

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


def _convert_config_to_mixtral(
    base_config: LlamaConfig | MistralConfig,
    num_experts: int,
    experts_per_token: Optional[int] = None,
) -> MixtralConfig:
    """
    Acknowledgement:
        This function is based on the implementation of Mixtral in the [mergekit](https://github.com/arcee-ai/mergekit) library.
        Mergekit is model merging library dedicated to large language models.

        Credit to https://github.com/arcee-ai/mergekit/blob/main/mergekit/moe/mixtral.py

    Args:
        base_config (LlamaConfig | MistralConfig): The base configuration of the model.
        num_experts (int): The number of experts in the Mixtral model.
        experts_per_token (Optional[int]): The number of experts per token. Defaults to 2.

    Returns:
        MixtralConfig: The configuration for the Mixtral model.
    """
    if not isinstance(base_config, MistralConfig):
        mistral_config = MistralConfig(**base_config.to_dict())
        mistral_config.sliding_window = None
        mistral_config.max_position_embeddings = base_config.max_position_embeddings
        base_config = mistral_config

    mixtral_config = MixtralConfig(**base_config.to_dict())
    mixtral_config.architectures = ["MixtralForCausalLM"]
    mixtral_config.num_local_experts = num_experts
    mixtral_config.num_experts_per_tok = experts_per_token or 2
    mixtral_config.sliding_window = None

    if (mixtral_config.num_local_experts & (mixtral_config.num_local_experts - 1)) != 0:
        logging.warning(
            f"Your model has {mixtral_config.num_local_experts} experts, which is "
            "not a power of two. The model will not be usable in llama.cpp."
        )
    return mixtral_config


def _convert_mlp(
    input_mlp: LlamaMLP | MistralMLP, output_mlp: MixtralBlockSparseTop2MLP
):
    """
    Convert the MLP layers from the input model to the output model.

    Args:
        input_mlp (LlamaMLP | MistralMLP): The MLP layer from the input model.
        output_mlp (MixtralBlockSparseTop2MLP): The MLP layer in the output model.

    Returns:
        None
    """
    output_mlp.w1.load_state_dict(input_mlp.gate_proj.state_dict())
    output_mlp.w2.load_state_dict(input_mlp.down_proj.state_dict())
    output_mlp.w3.load_state_dict(input_mlp.up_proj.state_dict())


def _upscale_decoder_layer(
    input_layer: LlamaDecoderLayer | MistralDecoderLayer,
    output_layer: MixtralDecoderLayer,
):
    """
    Copy the weights from the input layer to the output layer.
    This will modify the output layer in place.

    Args:
        input_layer (LlamaDecoderLayer | MistralDecoderLayer): The decoder layer from the input model.
        output_layer (MixtralDecoderLayer): The decoder layer in the output model.

    Returns:
        None
    """
    output_layer.input_layernorm.load_state_dict(
        input_layer.input_layernorm.state_dict()
    )
    output_layer.post_attention_layernorm.load_state_dict(
        input_layer.post_attention_layernorm.state_dict()
    )
    for expert in output_layer.block_sparse_moe.experts:
        _convert_mlp(input_layer.mlp, expert)


def upscale_to_mixtral_model(
    input_model: LlamaModel | MistralModel, output_model: MixtralModel
):
    """
    A helper function.

    Upscales a LlamaModel or MistralModel to a MixtralModel.

    Args:
        input_model (LlamaModel | MistralModel): The input model to be upscaled.
        output_model (MixtralModel): The output model where the upscaled weights will be loaded.

    Returns:
        None
    """
    # copy the weights from the pretrained model
    output_model.embed_tokens.load_state_dict(input_model.embed_tokens.state_dict())
    output_model.norm.load_state_dict(input_model.norm.state_dict())
    for input_layer, output_layer in tqdm(
        zip(input_model.layers, output_model.layers),
        desc="Upscaling layers",
        total=len(input_model.layers),
    ):
        _upscale_decoder_layer(input_layer, output_layer)


def upscale_to_mixtral_for_causal_lm(
    input_model: LlamaForCausalLM | MistralForCausalLM, output_model: MixtralForCausalLM
):
    """
    A helper function.

    Upscales a LlamaForCausalLM or MistralForCausalLM to a MixtralForCausalLM.

    Args:
        input_model (LlamaForCausalLM | MistralForCausalLM): The input model to be upscaled.
        output_model (MixtralForCausalLM): The output model where the upscaled weights will be loaded.

    Returns:
        None
    """
    output_model.lm_head.load_state_dict(input_model.lm_head.state_dict())
    upscale_to_mixtral_model(input_model.model, output_model.model)


class MixtralUpscalingAlgorithm(BaseAlgorithm):
    """
    This class is responsible for upscaling a model to a MixtralModel.
    It inherits from the ModelFusionAlgorithm class.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "num_experts": "num_experts",
        "experts_per_token": "experts_per_token",
        "save_checkpoint": "save_checkpoint",
    }

    def __init__(
        self,
        num_experts: int,
        experts_per_token: int,
        save_checkpoint: str,
        **kwargs,
    ):
        """
        Initialize the MixtralUpscalingAlgorithm.

        Args:
            num_experts (int): The number of experts in the Mixtral model.
            experts_per_token (int): The number of experts per token.
            save_checkpoint (str): The path to save the checkpoint.
            **kwargs: Additional keyword arguments.
        """
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.save_checkpoint = save_checkpoint
        super().__init__(**kwargs)

    @torch.no_grad()
    def _run(
        self, modelpool: BaseModelPool | LlamaModel | MistralModel
    ) -> MixtralModel:
        """
        Internal method to run the upscaling process.

        Args:
            modelpool (BaseModelPool | LlamaModel | MistralModel): The model to be upscaled.

        Returns:
            MixtralModel: The upscaled model.
        """
        if isinstance(modelpool, BaseModelPool):
            assert modelpool.has_pretrained, "ModelPool must have pretrained model."
            pretrained_model = modelpool.load_model("_pretrained_")
        elif isinstance(modelpool, (LlamaModel, MistralModel)):
            pretrained_model = modelpool
        else:
            raise ValueError("Invalid modelpool type")

        mixtral_config = _convert_config_to_mixtral(
            pretrained_model.config,
            self.config.num_experts,
            self.config.experts_per_token,
        )

        with ContextManagers([no_init_weights(True)]):
            for _ in tqdm(range(1), desc="Initializing Mixtral model"):
                mixtral_model = MixtralModel(mixtral_config)
        upscale_to_mixtral_model(pretrained_model, mixtral_model)

        return mixtral_model

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | LlamaModel | MistralModel) -> MixtralModel:
        """
        Runs the upscaling process.

        Args:
            modelpool (ModelPool | LlamaModel | MistralModel): The model to be upscaled.

        Returns:
            MixtralModel: The upscaled model.
        """
        mixtral_model = self._run(modelpool)

        if self.config.get("save_checkpoint", None) is not None:
            mixtral_model.save_pretrained(self.config.save_checkpoint)
        return mixtral_model


class MixtralForCausalLMUpscalingAlgorithm(BaseAlgorithm):
    """
    This class is responsible for upscaling a model to a MixtralForCausalLM.
    It inherits from the ModelFusionAlgorithm class.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "num_experts": "num_experts",
        "experts_per_token": "experts_per_token",
        "save_checkpoint": "save_checkpoint",
    }

    def __init__(
        self,
        num_experts: int,
        experts_per_token: int,
        save_checkpoint: str,
        **kwargs,
    ):
        """
        Initialize the MixtralForCausalLMUpscalingAlgorithm.

        Args:
            num_experts (int): The number of experts in the Mixtral model.
            experts_per_token (int): The number of experts per token.
            save_checkpoint (str): The path to save the checkpoint.
            **kwargs: Additional keyword arguments.
        """
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.save_checkpoint = save_checkpoint
        super().__init__(**kwargs)

    @torch.no_grad()
    def _run(
        self, modelpool: BaseModelPool | LlamaForCausalLM | MistralForCausalLM
    ) -> MixtralForCausalLM:
        """
        Internal method to run the upscaling process.

        Args:
            modelpool (BaseModelPool | LlamaForCausalLM | MistralForCausalLM): The model to be upscaled.

        Returns:
            MixtralForCausalLM: The upscaled model.
        """
        if isinstance(modelpool, BaseModelPool):
            assert modelpool.has_pretrained, "ModelPool must have pretrained model."
            pretrained_model = modelpool.load_model("_pretrained_")
        elif isinstance(modelpool, (LlamaForCausalLM, MistralForCausalLM)):
            pretrained_model = modelpool
        else:
            raise ValueError("Invalid modelpool type")

        mixtral_config = _convert_config_to_mixtral(
            pretrained_model.config,
            self.config.num_experts,
            self.config.experts_per_token,
        )

        with ContextManagers([no_init_weights(True)]):
            for _ in tqdm(range(1), desc="Initializing Mixtral model"):
                mixtral_model = MixtralForCausalLM(mixtral_config)
        upscale_to_mixtral_for_causal_lm(pretrained_model, mixtral_model)

        return mixtral_model

    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | LlamaForCausalLM | MistralForCausalLM
    ) -> MixtralForCausalLM:
        """
        Runs the upscaling process.

        Args:
            modelpool (ModelPool | LlamaForCausalLM | MistralForCausalLM): The model to be upscaled.

        Returns:
            MixtralForCausalLM: The upscaled model.
        """
        mixtral_model = self._run(modelpool)

        if self.config.get("save_checkpoint", None) is not None:
            mixtral_model.save_pretrained(self.config.save_checkpoint)
        return mixtral_model
