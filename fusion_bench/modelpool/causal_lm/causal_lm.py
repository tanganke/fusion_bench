import functools
import logging
import os
from typing import Optional, cast

from omegaconf import DictConfig, OmegaConf
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


def config_priority_get(priority_config, general_config, key, default):
    """
    Retrieve a configuration value with priority.

    This function retrieves the value associated with `key` from `priority_config` if it exists.
    If the key is not found in `priority_config`, it retrieves the value from `general_config`.
    If the key is not found in either configuration, it returns the provided `default` value.

    Args:
        priority_config (dict): The configuration dictionary with higher priority.
        general_config (dict): The general configuration dictionary.
        key (str): The key to look up in the configuration dictionaries.
        default: The default value to return if the key is not found in either configuration.

    Returns:
        The value associated with `key` from `priority_config` or `general_config`, or the `default` value if the key is not found.
    """
    if key in priority_config:
        return priority_config[key]
    return general_config.get(key, default)


class AutoModelForCausalLMPool(ModelPool):
    def load_model(self, model_config: str | DictConfig) -> Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        else:
            model_config = model_config

        config = self.config
        get_option = lambda key, default: config_priority_get(
            priority_config=model_config,
            general_config=config,
            key=key,
            default=default,
        )

        kwargs = get_option("model_kwargs", {})
        if isinstance(kwargs, DictConfig):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = parse_dtype(kwargs["torch_dtype"])
        if (
            "torch_dtype" not in kwargs
            and (dtype := get_option("dtype", None)) is not None
        ):
            kwargs["torch_dtype"] = parse_dtype(dtype)

        with timeit_context(f"loading model from {model_config.path}"):
            model = AutoModelForCausalLM.from_pretrained(
                os.path.expanduser(model_config.path), **kwargs
            )
        return model

    def load_tokenizer(self, model_config: str | DictConfig, **kwargs):
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        else:
            model_config = model_config

        with timeit_context(f"loading tokenizer from {model_config.path}"):
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.expanduser(model_config.path), **kwargs
            )
        return tokenizer

    def load_pretrained_or_first_tokenizer(self, **kwargs):
        if self.has_pretrained:
            return self.load_tokenizer("_pretrained_", **kwargs)
        return self.load_tokenizer(self.model_names[0], **kwargs)

    @override
    def save_model(
        self,
        model: PreTrainedModel,
        path: str,
        push_to_hub: bool = False,
        model_dtype: Optional[str] = None,
        save_tokenizer: bool = False,
        tokenizer_kwargs=None,
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
        if save_tokenizer:
            if tokenizer_kwargs is None:
                tokenizer_kwargs = {}
            # load the tokenizer
            if self.has_pretrained:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.expanduser(self.get_model_config("_pretrained_").path),
                    **tokenizer_kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.expanduser(
                        self.get_model_config(self.model_names[0]).path,
                        **tokenizer_kwargs,
                    )
                )
            tokenizer.save_pretrained(
                path,
                push_to_hub=push_to_hub,
            )
        if model_dtype is not None:
            model.to(dtype=parse_dtype(model_dtype))
        model.save_pretrained(
            path,
            push_to_hub=push_to_hub,
            **kwargs,
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
