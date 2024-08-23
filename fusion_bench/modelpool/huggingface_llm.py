import functools
import logging
import os
from typing import Optional, cast

from omegaconf import DictConfig
from torch.nn.modules import Module
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM,
    PreTrainedModel,
)
from typing_extensions import override

from fusion_bench.modelpool.base_pool import ModelPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.dtype import parse_dtype

log = logging.getLogger(__name__)


class AutoModelForCausalLMPool(ModelPool):
    def load_model(self, model_config: str | DictConfig) -> Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        else:
            model_config = model_config

        kwargs = {}
        if self.config.get("dtype", None) is not None:
            kwargs["torch_dtype"] = parse_dtype(self.config.dtype)

        with timeit_context(f"loading model from {model_config.path}"):
            model = AutoModelForCausalLM.from_pretrained(
                os.path.expanduser(model_config.path), **kwargs
            )
        return model

    @override
    def save_model(
        self,
        model: PreTrainedModel,
        path: str,
        push_to_hub: bool = False,
        save_tokenizer: bool = False,
        **kwargs,
    ):
        """
        Save the model to the specified path.

        Args:
            model (PreTrainedModel): The model to be saved.
            path (str): The path where the model will be saved.
            push_to_hub (bool, optional): Whether to push the model to the Hugging Face Hub. Defaults to False.
            save_tokenizer (bool, optional): Whether to save the tokenizer along with the model. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `save_pretrained` method.
        """
        path = os.path.expanduser(path)
        model.save_pretrained(
            path,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        if save_tokenizer:
            if self.has_pretrained:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.expanduser(self.get_model_config("_pretrained_").path)
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.expanduser(self.get_model_config(self.model_names[0]).path)
                )
            tokenizer.save_pretrained(
                path,
                push_to_hub=push_to_hub,
            )


class LLamaForCausalLMPool(AutoModelForCausalLMPool):
    @override
    def load_model(
        self,
        model_config: str | DictConfig,
        backbone_only: bool = False,
    ):
        model = super().load_model(model_config)
        model = cast(LlamaForCausalLM, model)
        if backbone_only:
            model = model.model
        return model


class MistralForCausalLMPool(AutoModelForCausalLMPool):
    @override
    def load_model(
        self,
        model_config: str | DictConfig,
        backbone_only: bool = False,
    ):
        model = super().load_model(model_config)
        model = cast(MistralForCausalLM, model)
        if backbone_only:
            model = model.model
        return model
