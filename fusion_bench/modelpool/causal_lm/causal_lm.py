"""
Online documentation for this module: https://tanganke.github.io/fusion_bench/modelpool/causal_lm
"""

import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional, TypeAlias, Union, cast  # noqa: F401

import peft
from omegaconf import DictConfig, flag_override
from torch import nn
from torch.nn.modules import Module
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils import instantiate
from fusion_bench.utils.dtype import parse_dtype

log = logging.getLogger(__name__)


class CausalLMPool(BaseModelPool):
    _config_mapping = BaseModelPool._config_mapping | {
        "_tokenizer": "tokenizer",
        "_model_kwargs": "model_kwargs",
    }

    def __init__(
        self,
        models,
        *,
        tokenizer: Optional[DictConfig],
        model_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        # process `model_kwargs`
        self._tokenizer = tokenizer
        self._model_kwargs = model_kwargs
        if self._model_kwargs is None:
            self._model_kwargs = DictConfig({})
        with flag_override(self._model_kwargs, "allow_objects", True):
            if hasattr(self._model_kwargs, "torch_dtype"):
                self._model_kwargs.torch_dtype = parse_dtype(
                    self._model_kwargs.torch_dtype
                )

    @override
    def load_model(
        self,
        model_name_or_config: str | DictConfig,
        *args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Example of YAML config:

        ```yaml
        models:
          _pretrained_: path_to_pretrained_model # if a plain string, it will be passed to AutoModelForCausalLM.from_pretrained
          model_a: path_to_model_a
          model_b: path_to_model_b
        ```

        or equivalently,

        ```yaml
        models:
          _pretrained_:
            _target_: transformers.AutoModelForCausalLM # any callable that returns a model
            pretrained_model_name_or_path: path_to_pretrained_model
          model_a:
            _target_: transformers.AutoModelForCausalLM
            pretrained_model_name_or_path: path_to_model_a
          model_b:
            _target_: transformers.AutoModelForCausalLM
            pretrained_model_name_or_path: path_to_model_b
        ```
        """
        model_kwargs = deepcopy(self._model_kwargs)
        model_kwargs.update(kwargs)

        if isinstance(model_name_or_config, str):
            log.info(f"Loading model: {model_name_or_config}", stacklevel=2)
            if model_name_or_config in self._models.keys():
                model_config = self._models[model_name_or_config]
                if isinstance(model_config, str):
                    # model_config is a string
                    model = AutoModelForCausalLM.from_pretrained(
                        model_config,
                        *args,
                        **model_kwargs,
                    )
                    return model
        elif isinstance(model_name_or_config, (DictConfig, Dict)):
            model_config = model_name_or_config

        model = instantiate(model_config, *args, **model_kwargs)
        return model

    def load_tokenizer(self, *args, **kwargs) -> PreTrainedTokenizer:
        """
        Example of YAML config:

        ```yaml
        tokenizer: google/gemma-2-2b-it # if a plain string, it will be passed to AutoTokenizer.from_pretrained
        ```

        or equivalently,

        ```yaml
        tokenizer:
          _target_: transformers.AutoTokenizer # any callable that returns a tokenizer
          pretrained_model_name_or_path: google/gemma-2-2b-it
        ```

        Returns:
            PreTrainedTokenizer: The tokenizer.
        """
        assert self._tokenizer is not None, "Tokenizer is not defined in the config"
        log.info("Loading tokenizer.", stacklevel=2)
        if isinstance(self._tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer, *args, **kwargs)
        else:
            tokenizer = instantiate(self._tokenizer, *args, **kwargs)
        return tokenizer

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
            tokenizer = self.load_tokenizer(**tokenizer_kwargs)
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


class CausalLMBackbonePool(CausalLMPool):
    def load_model(
        self, model_name_or_config: str | DictConfig, *args, **kwargs
    ) -> Module:
        model: AutoModelForCausalLM = super().load_model(
            model_name_or_config, *args, **kwargs
        )
        return model.model.layers


def load_peft_causal_lm(
    base_model_path: str,
    peft_model_path: str,
    torch_dtype: str = "bfloat16",
    is_trainable: bool = True,
    merge_and_unload: bool = False,
):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch_dtype
    )
    model = peft.PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        is_trainable=is_trainable,
    )
    if merge_and_unload:
        model = model.merge_and_unload()
    return model
