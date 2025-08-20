import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import torch
from accelerate import init_empty_weights
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.constants import RuntimeConstants
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.models.hf_utils import (
    create_default_model_card,
    save_pretrained_with_remote_code,
)
from fusion_bench.models.modeling_smile_llama import (
    SmileLlamaConfig,
    SmileLlamaForCausalLM,
    SmileLlamaModel,
)
from fusion_bench.models.modeling_smile_llama.modeling_smile_llama import (
    SmileLlamaDecoderLayer,
)
from fusion_bench.models.modeling_smile_mistral import (
    SmileMistralConfig,
    SmileMistralForCausalLM,
    SmileMistralModel,
)
from fusion_bench.models.modeling_smile_mistral.modeling_smile_mistral import (
    SmileMistralDecoderLayer,
)

# Import all SMILE configurations and models
from fusion_bench.models.modeling_smile_qwen2 import (
    SmileQwen2Config,
    SmileQwen2ForCausalLM,
    SmileQwen2Model,
)
from fusion_bench.models.modeling_smile_qwen2.modeling_smile_qwen2 import (
    SmileQwen2DecoderLayer,
)
from fusion_bench.models.smile_moe.linear_from_hf_config import (
    ExpertNotTrainedError,
    upscale_to_smile_linear,
)
from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)

# Model type mappings
MODEL_TYPE_MAPPINGS = {
    "qwen2": {
        "base_model_cls": Qwen2ForCausalLM,
        "base_decoder_layer_cls": Qwen2DecoderLayer,
        "smile_config_cls": SmileQwen2Config,
        "smile_model_cls": SmileQwen2ForCausalLM,
        "smile_base_model_cls": SmileQwen2Model,
        "smile_decoder_layer_cls": SmileQwen2DecoderLayer,
        "description": "Qwen2",
    },
    "llama": {
        "base_model_cls": LlamaForCausalLM,
        "base_decoder_layer_cls": LlamaDecoderLayer,
        "smile_config_cls": SmileLlamaConfig,
        "smile_model_cls": SmileLlamaForCausalLM,
        "smile_base_model_cls": SmileLlamaModel,
        "smile_decoder_layer_cls": SmileLlamaDecoderLayer,
        "description": "Llama",
    },
    "mistral": {
        "base_model_cls": MistralForCausalLM,
        "base_decoder_layer_cls": MistralDecoderLayer,
        "smile_config_cls": SmileMistralConfig,
        "smile_model_cls": SmileMistralForCausalLM,
        "smile_base_model_cls": SmileMistralModel,
        "smile_decoder_layer_cls": SmileMistralDecoderLayer,
        "description": "Mistral",
    },
}


def detect_model_type(
    model_or_config: Union[PreTrainedModel, PretrainedConfig, str],
) -> str:
    """
    Detect the model type from a model, config, or model name/path.

    Args:
        model_or_config: Model, config, or model name/path to detect type from

    Returns:
        str: The detected model type ("qwen2", "llama", "mistral")

    Raises:
        ValueError: If model type cannot be detected or is not supported
    """
    if isinstance(model_or_config, str):
        # Load config from path/name
        config = AutoConfig.from_pretrained(model_or_config)
    elif isinstance(model_or_config, PreTrainedModel):
        config = model_or_config.config
    elif isinstance(model_or_config, PretrainedConfig):
        config = model_or_config
    else:
        raise ValueError(
            f"Unsupported type for model type detection: {type(model_or_config)}"
        )

    model_type = getattr(config, "model_type", "").lower()

    # Handle various model type variations
    if model_type in MODEL_TYPE_MAPPINGS:
        return model_type
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: {list(MODEL_TYPE_MAPPINGS.keys())}"
        )


@auto_register_config
class SmileCausalLMUpscalingAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    R"""
    SmileCausalLMUpscalingAlgorithm is a generic model fusion algorithm designed to upscale
    a pretrained CausalLM model using a set of fine-tuned expert models. The algorithm
    supports Qwen2, Llama, and Mistral model architectures and leverages Singular Value
    Decomposition (SVD) to merge the weights of the pretrained model and the expert models
    into a new upscaled model.

    The algorithm automatically detects the model type and uses the appropriate SMILE
    configuration and model classes.

    Methods:
        run(modelpool: BaseModelPool) -> Union[SmileQwen2ForCausalLM, SmileLlamaForCausalLM, SmileMistralForCausalLM]:
            Executes the upscaling process and returns the upscaled model.

        merge(pretrained_model: PreTrainedModel, finetuned_models: List[PreTrainedModel]) -> PreTrainedModel:
            Merges the pretrained model with the fine-tuned models to create an upscaled model.
    """

    modelpool: CausalLMPool

    def __init__(
        self,
        device,
        accelerator,
        model_save_path,
        model_dtype,
        num_experts_per_tok,
        rank_of_router,
        rank_of_expert,
        save_with_remote_code: bool = True,
        model_type: str = None,  # Optional: explicitly specify model type
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_mappings = None  # Will be set during run()

        if not torch.cuda.is_available():
            if "cuda" in self.device:
                self.device = "cpu"
            if "cuda" in self.accelerator:
                self.accelerator = "cpu"

    @torch.no_grad()
    def run(self, modelpool) -> PreTrainedModel:
        """
        Executes the upscaling process.

        Args:
            modelpool (ModelPool): The pool of models to be used for upscaling.

        Returns:
            PreTrainedModel: The upscaled model (specific type depends on detected model architecture).
        """
        self.modelpool = modelpool = to_modelpool(modelpool)
        config = self.config

        # Auto-detect model type if not specified
        if self.model_type is None:
            self.model_type = detect_model_type(
                modelpool.get_model_path("_pretrained_")
            )
            log.info(f"Auto-detected model type: {self.model_type}")

        # Get the appropriate model mappings
        if self.model_type not in MODEL_TYPE_MAPPINGS:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. Supported: {list(MODEL_TYPE_MAPPINGS.keys())}"
            )

        self.model_mappings = MODEL_TYPE_MAPPINGS[self.model_type]
        log.info(f"Using {self.model_mappings['description']} model architecture")

        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_pretrained_model()

        with self.profile("load fine-tuned model"):
            finetuned_models = [
                m for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
            ]

        if self.device == "cuda" and torch.cuda.is_available():
            pretrained_model = pretrained_model.cuda()
            print("parameter count of pretrained model:")
            print_parameters(pretrained_model)
            finetuned_models = [m.cuda() for m in finetuned_models]

        with self.profile("merge model"):
            model = self.merge(pretrained_model, finetuned_models)

        self.print_profile_summary()
        print("parameter count of upscaled MoE model:")
        print_parameters(model)
        print(model)

        if self.model_dtype is not None:
            model.to(dtype=parse_dtype(self.model_dtype))

        if self.model_save_path is not None:
            if os.path.dirname(self.model_save_path):
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            log.info(f"Saving model to {self.model_save_path}")
            tokenizer = self.modelpool.load_tokenizer()
            tokenizer.save_pretrained(self.model_save_path)
            if not self.save_with_remote_code:
                model.save_pretrained(self.model_save_path)
            else:
                # Use the appropriate auto_map for the detected model type
                auto_map = {
                    "AutoConfig": self.model_mappings["smile_config_cls"],
                    "AutoModel": self.model_mappings["smile_base_model_cls"],
                    "AutoModelForCausalLM": self.model_mappings["smile_model_cls"],
                }
                save_pretrained_with_remote_code(
                    model,
                    auto_map=auto_map,
                    save_directory=self.model_save_path,
                )

            # save readme
            model_card_str = create_default_model_card(
                models=[modelpool.get_model_path(m) for m in modelpool.all_model_names],
                description=f"Merged {self.model_mappings['description']} model using SMILE Upscaling",
                algorithm_config=self.config,
                modelpool_config=modelpool.config,
            )
            with open(os.path.join(self.model_save_path, "README.md"), "w") as f:
                f.write(model_card_str)

        return model

    def merge(
        self,
        pretrained_model: PreTrainedModel,
        finetuned_models: List[PreTrainedModel],
    ) -> PreTrainedModel:
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            pretrained_model (PreTrainedModel): The pretrained model.
            finetuned_models (List[PreTrainedModel]): A list of fine-tuned models.

        Returns:
            PreTrainedModel: The upscaled model (specific type depends on model architecture).
        """
        with init_empty_weights():
            pretrained_model_config = self.modelpool.get_model_config("_pretrained_")
            if isinstance(pretrained_model_config, str):
                pretrained_path = pretrained_model_config
            else:
                pretrained_path = pretrained_model_config.get(
                    "path", pretrained_model_config["pretrained_model_name_or_path"]
                )
            base_config = AutoConfig.from_pretrained(pretrained_path)

            # Create the appropriate SMILE config for the detected model type
            SmileConfigClass = self.model_mappings["smile_config_cls"]
            model_config = SmileConfigClass(
                num_experts_per_tok=self.num_experts_per_tok,
                rank_of_router=self.rank_of_router,
                rank_of_expert=self.rank_of_expert,
                num_local_experts=len(finetuned_models),
                **base_config.to_dict(),
            )

            # Create the appropriate SMILE model for the detected model type
            SmileModelClass = self.model_mappings["smile_model_cls"]
            model = SmileModelClass(model_config)

        model.to(dtype=pretrained_model.dtype).to_empty(device="cpu")

        # copy pretrained model weights
        state_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        for key in list(pretrained_state_dict.keys()):
            if key not in state_dict:
                pretrained_state_dict.pop(key)
        model.load_state_dict(pretrained_state_dict, strict=False)

        # upscale model
        BaseDecoderLayerClass = self.model_mappings["base_decoder_layer_cls"]
        SmileDecoderLayerClass = self.model_mappings["smile_decoder_layer_cls"]

        for layer_idx in tqdm(
            range(len(pretrained_model.model.layers)),
            "Upscaling Modules (layer)",
            dynamic_ncols=True,
        ):
            if RuntimeConstants.debug and layer_idx > 0:
                log.info(
                    "Debug mode enabled: processing only the first layer, skipping remaining layers"
                )
                break

            pretrained_layer = pretrained_model.model.layers[layer_idx]
            finetuned_layers = [m.model.layers[layer_idx] for m in finetuned_models]

            target_layer = model.model.layers[layer_idx]

            for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                try:
                    upscale_to_smile_linear(
                        base=getattr(pretrained_layer.self_attn, n),
                        experts=[getattr(m.self_attn, n) for m in finetuned_layers],
                        target=getattr(target_layer.self_attn, n),
                        accelerator=self.accelerator,
                    )
                except ExpertNotTrainedError:
                    setattr(
                        target_layer.self_attn,
                        n,
                        getattr(pretrained_layer.self_attn, n),
                    )

            for n in ["gate_proj", "up_proj", "down_proj"]:
                try:
                    upscale_to_smile_linear(
                        base=getattr(pretrained_layer.mlp, n),
                        experts=[getattr(m.mlp, n) for m in finetuned_layers],
                        target=getattr(target_layer.mlp, n),
                        accelerator=self.accelerator,
                    )
                except ExpertNotTrainedError:
                    setattr(
                        target_layer.mlp,
                        n,
                        getattr(pretrained_layer.mlp, n),
                    )

        return model
