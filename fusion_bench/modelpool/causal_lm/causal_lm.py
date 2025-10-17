"""
Online documentation for this module: https://tanganke.github.io/fusion_bench/modelpool/llm
"""

import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional, TypeAlias, Union, cast  # noqa: F401

import peft
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, flag_override
from torch import nn
from torch.nn.modules import Module
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from fusion_bench import (
    BaseModelPool,
    auto_register_config,
    import_object,
    instantiate,
    parse_dtype,
)
from fusion_bench.models.hf_utils import create_default_model_card
from fusion_bench.utils.lazy_state_dict import LazyStateDict

log = logging.getLogger(__name__)


@auto_register_config
class CausalLMPool(BaseModelPool):
    """A model pool for managing and loading causal language models.

    This class provides a unified interface for loading and managing multiple
    causal language models, typically used in model fusion and ensemble scenarios.
    It supports both eager and lazy loading strategies, and handles model
    configuration through YAML configs or direct instantiation.

    The pool can manage models from Hugging Face Hub, local paths, or custom
    configurations. It also provides tokenizer management and model saving
    capabilities with optional Hugging Face Hub integration.

    Args:
        models: Dictionary or configuration specifying the models to be managed.
            Can contain model names mapped to paths or detailed configurations.
        tokenizer: Tokenizer configuration, either a string path/name or
            a DictConfig with detailed tokenizer settings.
        model_kwargs: Additional keyword arguments passed to model loading.
            Common options include torch_dtype, device_map, etc.
        enable_lazy_loading: Whether to use lazy loading for models. When True,
            models are loaded as LazyStateDict objects instead of actual models,
            which can save memory for large model collections.
        **kwargs: Additional arguments passed to the parent BaseModelPool.

    Example:
        ```python
        >>> pool = CausalLMPool(
        ...     models={
        ...         "model_a": "microsoft/DialoGPT-medium",
        ...         "model_b": "/path/to/local/model"
        ...     },
        ...     tokenizer="microsoft/DialoGPT-medium",
        ...     model_kwargs={"torch_dtype": "bfloat16"}
        ... )
        >>> model = pool.load_model("model_a")
        >>> tokenizer = pool.load_tokenizer()
        ```
    """

    def __init__(
        self,
        models,
        *,
        tokenizer: Optional[DictConfig | str],
        model_kwargs: Optional[DictConfig] = None,
        enable_lazy_loading: bool = False,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        if model_kwargs is None:
            self.model_kwargs = DictConfig({})

    def get_model_path(self, model_name: str):
        """Extract the model path from the model configuration.

        Args:
            model_name: The name of the model as defined in the models configuration.

        Returns:
            str: The path or identifier for the model. For string configurations,
                returns the string directly. For dict configurations, extracts
                the 'pretrained_model_name_or_path' field.

        Raises:
            RuntimeError: If the model configuration is invalid or the model
                name is not found in the configuration.
        """
        model_name_or_config = self._models[model_name]
        if isinstance(model_name_or_config, str):
            return model_name_or_config
        elif isinstance(model_name_or_config, (DictConfig, dict)):
            return model_name_or_config.get("pretrained_model_name_or_path")
        else:
            raise RuntimeError("Invalid model configuration")

    def get_model_kwargs(self):
        """Get processed model keyword arguments for model loading.

        Converts the stored `model_kwargs` from DictConfig to a regular dictionary
        and processes special arguments like torch_dtype for proper model loading.

        Returns:
            dict: Processed keyword arguments ready to be passed to model
                loading functions. The torch_dtype field, if present, is
                converted from string to the appropriate torch dtype object.
        """
        model_kwargs = (
            OmegaConf.to_container(self.model_kwargs, resolve=True)
            if isinstance(self.model_kwargs, DictConfig)
            else self.model_kwargs
        )
        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = parse_dtype(model_kwargs["torch_dtype"])
        return model_kwargs

    @override
    def load_model(
        self,
        model_name_or_config: str | DictConfig,
        *args,
        **kwargs,
    ) -> Union[PreTrainedModel, LazyStateDict]:
        """Load a causal language model from the model pool.

        This method supports multiple loading strategies:
        1. Loading by model name from the configured model pool
        2. Loading from a direct configuration dictionary
        3. Lazy loading using LazyStateDict for memory efficiency

        The method automatically handles different model configuration formats
        and applies the appropriate loading strategy based on the enable_lazy_loading flag.

        Args:
            model_name_or_config: Either a string model name that exists in the
                model pool configuration, or a DictConfig/dict containing the
                model configuration directly.
            *args: Additional positional arguments passed to the model constructor.
            **kwargs: Additional keyword arguments passed to the model constructor.
                These will be merged with the pool's model_kwargs.

        Returns:
            Union[PreTrainedModel, LazyStateDict]: The loaded model. Returns a
                PreTrainedModel for normal loading or a LazyStateDict for lazy loading.

        Raises:
            RuntimeError: If the model configuration is invalid.
            KeyError: If the model name is not found in the model pool.

        Example YAML configurations:
            Simple string configuration:
            ```yaml
            models:
              _pretrained_: path_to_pretrained_model
              model_a: path_to_model_a
              model_b: path_to_model_b
            ```

            Detailed configuration:
            ```yaml
            models:
              _pretrained_:
                _target_: transformers.AutoModelForCausalLM
                pretrained_model_name_or_path: path_to_pretrained_model
              model_a:
                _target_: transformers.AutoModelForCausalLM
                pretrained_model_name_or_path: path_to_model_a
            ```
        """
        model_kwargs = self.get_model_kwargs()
        model_kwargs.update(kwargs)

        if isinstance(model_name_or_config, str):
            # If model_name_or_config is a string, it is the name or the path of the model
            log.info(f"Loading model: {model_name_or_config}", stacklevel=2)
            if model_name_or_config in self._models.keys():
                model_config = self._models[model_name_or_config]
                if isinstance(model_config, str):
                    # model_config is a string
                    if not self.enable_lazy_loading:
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

        if not self.enable_lazy_loading:
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
        """Load the tokenizer associated with this model pool.

        Loads a tokenizer based on the tokenizer configuration provided during
        pool initialization. Supports both simple string paths and detailed
        configuration dictionaries.

        Args:
            *args: Additional positional arguments passed to the tokenizer constructor.
            **kwargs: Additional keyword arguments passed to the tokenizer constructor.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer instance.

        Raises:
            AssertionError: If no tokenizer is defined in the configuration.

        Example YAML configurations:
            Simple string configuration:
            ```yaml
            tokenizer: google/gemma-2-2b-it
            ```

            Detailed configuration:
            ```yaml
            tokenizer:
              _target_: transformers.AutoTokenizer
              pretrained_model_name_or_path: google/gemma-2-2b-it
              use_fast: true
              padding_side: left
            ```
        """
        assert self.tokenizer is not None, "Tokenizer is not defined in the config"
        log.info("Loading tokenizer.", stacklevel=2)
        if isinstance(self.tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, *args, **kwargs)
        else:
            tokenizer = instantiate(self.tokenizer, *args, **kwargs)
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
        algorithm_config: Optional[DictConfig] = None,
        description: Optional[str] = None,
        base_model_in_modelcard: bool = True,
        **kwargs,
    ):
        """Save a model to the specified path with optional tokenizer and Hub upload.

        This method provides comprehensive model saving capabilities including
        optional tokenizer saving, dtype conversion, model card creation, and
        Hugging Face Hub upload. The model is saved in the standard Hugging Face format.

        Args:
            model: The PreTrainedModel instance to be saved.
            path: The local path where the model will be saved. Supports tilde
                expansion for home directory paths.
            push_to_hub: Whether to push the saved model to the Hugging Face Hub.
                Requires proper authentication and repository permissions.
            model_dtype: Optional string specifying the target dtype for the model
                before saving (e.g., "float16", "bfloat16"). The model will be
                converted to this dtype before saving.
            save_tokenizer: Whether to save the tokenizer alongside the model.
                If True, the tokenizer will be loaded using the pool's tokenizer
                configuration and saved to the same path.
            tokenizer_kwargs: Additional keyword arguments for tokenizer loading
                when save_tokenizer is True.
            tokenizer: Optional pre-loaded tokenizer instance. If provided, this
                tokenizer will be saved regardless of the save_tokenizer flag.
            algorithm_config: Optional DictConfig containing algorithm configuration.
                If provided, a model card will be created with algorithm details.
            description: Optional description for the model card. If not provided
                and algorithm_config is given, a default description will be generated.
            **kwargs: Additional keyword arguments passed to the model's
                save_pretrained method.

        Example:
            ```python
            >>> pool = CausalLMPool(models=..., tokenizer=...)
            >>> model = pool.load_model("my_model")
            >>> pool.save_model(
            ...     model,
            ...     "/path/to/save",
            ...     save_tokenizer=True,
            ...     model_dtype="float16",
            ...     push_to_hub=True,
            ...     algorithm_config=algorithm_config,
            ...     description="Custom merged model"
            ... )
            ```
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

        # Create and save model card if algorithm_config is provided
        if algorithm_config is not None and rank_zero_only.rank == 0:
            if description is None:
                description = "Model created using FusionBench."
            model_card_str = create_default_model_card(
                base_model=(
                    self.get_model_path("_pretrained_")
                    if base_model_in_modelcard and self.has_pretrained
                    else None
                ),
                models=[self.get_model_path(m) for m in self.model_names],
                description=description,
                algorithm_config=algorithm_config,
                modelpool_config=self.config,
            )
            with open(os.path.join(path, "README.md"), "w") as f:
                f.write(model_card_str)


class CausalLMBackbonePool(CausalLMPool):
    """A specialized model pool that loads only the transformer backbone layers.

    This class extends CausalLMPool to provide access to just the transformer
    layers (backbone) of causal language models, excluding the language modeling
    head and embeddings. This is useful for model fusion scenarios where only
    the core transformer layers are needed.

    The class automatically extracts the `model.layers` component from loaded
    AutoModelForCausalLM instances, providing direct access to the transformer
    blocks. Lazy loading is not supported for this pool type.

    Note:
        This pool automatically disables lazy loading as it needs to access
        the internal structure of the model to extract the backbone layers.

    Example:
        ```python
        >>> backbone_pool = CausalLMBackbonePool(
        ...     models={"model_a": "microsoft/DialoGPT-medium"},
        ...     tokenizer="microsoft/DialoGPT-medium"
        ... )
        >>> layers = backbone_pool.load_model("model_a")  # Returns nn.ModuleList of transformer layers
        ```
    """

    def load_model(
        self, model_name_or_config: str | DictConfig, *args, **kwargs
    ) -> Module:
        """Load only the transformer backbone layers from a causal language model.

        This method loads a complete causal language model and then extracts
        only the transformer layers (backbone), discarding the embedding layers
        and language modeling head. This is useful for model fusion scenarios
        where only the core transformer computation is needed.

        Args:
            model_name_or_config: Either a string model name from the pool
                configuration or a DictConfig with model loading parameters.
            *args: Additional positional arguments passed to the parent load_model method.
            **kwargs: Additional keyword arguments passed to the parent load_model method.

        Returns:
            Module: The transformer layers (typically a nn.ModuleList) containing
                the core transformer blocks without embeddings or output heads.

        Note:
            Lazy loading is automatically disabled for this method as it needs
            to access the internal model structure to extract the layers.
        """
        if self.enable_lazy_loading:
            log.warning(
                "CausalLMBackbonePool does not support lazy loading. "
                "Falling back to normal loading."
            )
            self.enable_lazy_loading = False
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
    """Load a causal language model with PEFT (Parameter-Efficient Fine-Tuning) adapters.

    This function loads a base causal language model and applies PEFT adapters
    (such as LoRA, AdaLoRA, or other parameter-efficient fine-tuning methods)
    to create a fine-tuned model. It supports both keeping the adapters separate
    or merging them into the base model.

    Args:
        base_model_path: Path or identifier for the base causal language model.
            Can be a Hugging Face model name or local path.
        peft_model_path: Path to the PEFT adapter configuration and weights.
            This should contain the adapter_config.json and adapter weights.
        torch_dtype: The torch data type to use for the model. Common options
            include "float16", "bfloat16", "float32". Defaults to "bfloat16".
        is_trainable: Whether the loaded PEFT model should be trainable.
            Set to False for inference-only usage to save memory.
        merge_and_unload: Whether to merge the PEFT adapters into the base model
            and unload the adapter weights. When True, returns a standard
            PreTrainedModel instead of a PeftModel.

    Returns:
        Union[PeftModel, PreTrainedModel]: The loaded model with PEFT adapters.
            Returns a PeftModel if merge_and_unload is False, or a PreTrainedModel
            if the adapters are merged and unloaded.

    Example:
        ```python
        >>> # Load model with adapters for training
        >>> model = load_peft_causal_lm(
        ...     "microsoft/DialoGPT-medium",
        ...     "/path/to/lora/adapters",
        ...     is_trainable=True
        ... )

        >>> # Load and merge adapters for inference
        >>> merged_model = load_peft_causal_lm(
        ...     "microsoft/DialoGPT-medium",
        ...     "/path/to/lora/adapters",
        ...     merge_and_unload=True,
        ...     is_trainable=False
        ... )
        ```
    """
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
