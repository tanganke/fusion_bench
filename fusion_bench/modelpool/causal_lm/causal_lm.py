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
from fusion_bench.utils.lazy_state_dict import LazyStateDict
from fusion_bench.utils.packages import import_object

log = logging.getLogger(__name__)


class CausalLMPool(BaseModelPool):
    _config_mapping = BaseModelPool._config_mapping | {
        "_tokenizer": "tokenizer",
        "_model_kwargs": "model_kwargs",
        "load_lazy": "load_lazy",
    }

    def __init__(
        self,
        models,
        *,
        tokenizer: Optional[DictConfig],
        model_kwargs: Optional[DictConfig] = None,
        load_lazy: bool = False,
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
        self.load_lazy = load_lazy

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
            # If model_name_or_config is a string, it is the name or the path of the model
            log.info(f"Loading model: {model_name_or_config}", stacklevel=2)
            if model_name_or_config in self._models.keys():
                model_config = self._models[model_name_or_config]
                if isinstance(model_config, str):
                    # model_config is a string
                    if not self.load_lazy:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_config,
                            *args,
                            **model_kwargs,
                        )
                    else:
                        # model_config is a string, but we want to use LazyStateDict
                        model = LazyStateDict(
                            checkpoint=model_config,
                            meta_module_class=AutoModelForCausalLM,
                            *args,
                            **model_kwargs,
                        )
                    return model
        elif isinstance(model_name_or_config, (DictConfig, Dict)):
            model_config = model_name_or_config

        if not self.load_lazy:
            model = instantiate(model_config, *args, **model_kwargs)
        else:
            meta_module_class = model_config.pop("_target_")
            checkpoint = model_config.pop("pretrained_model_name_or_path")
            model = LazyStateDict(
                checkpoint=checkpoint,
                meta_module_class=meta_module_class,
                *args,
                **model_kwargs,
            )
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
        tokenizer: Optional[PreTrainedTokenizer] = None,
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
        # NOTE: if tokenizer is provided, it will be saved regardless of `save_tokenizer`
        if save_tokenizer or tokenizer is not None:
            if tokenizer is None:
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
        if self.load_lazy:
            log.warning(
                "CausalLMBackbonePool does not support lazy loading. "
                "Falling back to normal loading."
            )
            self.load_lazy = False
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
