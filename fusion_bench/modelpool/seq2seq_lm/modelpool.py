import logging
from copy import deepcopy
from typing import Optional

from omegaconf import DictConfig, flag_override
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils import parse_dtype

log = logging.getLogger(__name__)


def load_lora_model(
    base_model_path: str,
    peft_model_path: str,
    is_trainable: bool = True,
    merge_and_unload: bool = True,
):
    """Load a sequence-to-sequence model with LoRA (Low-Rank Adaptation) fine-tuning.

    This function loads a base sequence-to-sequence language model and applies
    LoRA adapters for parameter-efficient fine-tuning. LoRA allows for efficient
    adaptation of large models by adding trainable low-rank matrices to the
    existing weights without modifying the original parameters.

    Args:
        base_model_path: Path or identifier for the base sequence-to-sequence model.
            Can be a Hugging Face model name (e.g., "t5-base") or local path.
        peft_model_path: Path to the directory containing LoRA adapter weights
            and configuration. Should include adapter_config.json and adapter weights.
        is_trainable: Whether the loaded model should be trainable. Set to False
            for inference-only usage to save memory and computation.
        merge_and_unload: Whether to merge the LoRA weights into the base model
            and unload the adapter. When True, returns a standard model instead
            of a PeftModel, which can be more efficient for inference.

    Returns:
        Union[PeftModel, AutoModelForSeq2SeqLM]: The loaded model with LoRA
            adapters. Returns a PeftModel if merge_and_unload is False, or
            a standard AutoModelForSeq2SeqLM if adapters are merged.

    Example:
        ```python
        >>> # Load model with separate adapters for training
        >>> model = load_lora_model(
        ...     "t5-base",
        ...     "/path/to/lora/adapters",
        ...     is_trainable=True,
        ...     merge_and_unload=False
        ... )

        >>> # Load and merge adapters for efficient inference
        >>> merged_model = load_lora_model(
        ...     "t5-base",
        ...     "/path/to/lora/adapters",
        ...     is_trainable=False,
        ...     merge_and_unload=True
        ... )
        ```
    """
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        is_trainable=is_trainable,
    )
    if merge_and_unload:
        model = model.merge_and_unload()
    return model


class Seq2SeqLMPool(BaseModelPool):
    """A model pool specialized for sequence-to-sequence language models.

    This model pool provides management and loading capabilities for sequence-to-sequence
    (seq2seq) language models such as T5, BART, and mT5. It extends the base model pool
    functionality with seq2seq-specific features including tokenizer management and
    model configuration handling.

    Seq2seq models are particularly useful for tasks that require generating output
    sequences from input sequences, such as translation, summarization, question
    answering, and text generation. This pool streamlines the process of loading
    and configuring multiple seq2seq models for fusion and ensemble scenarios.

    Key Features:
        - Specialized loading for AutoModelForSeq2SeqLM models
        - Integrated tokenizer management
        - Support for model-specific keyword arguments
        - Automatic dtype parsing and configuration
        - Compatible with PEFT (Parameter-Efficient Fine-Tuning) adapters

    Attributes:
        _tokenizer: Configuration for the tokenizer associated with the models
        _model_kwargs: Default keyword arguments applied to all model loading operations

    Example:
        ```python
        pool = Seq2SeqLMPool(
            models={
                "t5_base": "t5-base",
                "t5_large": "t5-large",
                "custom_model": "/path/to/local/model"
            },
            tokenizer={"_target_": "transformers.T5Tokenizer",
                      "pretrained_model_name_or_path": "t5-base"},
            model_kwargs={"torch_dtype": "float16", "device_map": "auto"}
        )
        model = pool.load_model("t5_base")
        tokenizer = pool.load_tokenizer()
        ```
    """

    _config_mapping = BaseModelPool._config_mapping | {
        "_tokenizer": "tokenizer",
        "_model_kwargs": "model_kwargs",
    }

    def __init__(
        self,
        models: DictConfig,
        *,
        tokenizer: Optional[DictConfig],
        model_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        """Initialize the sequence-to-sequence language model pool.

        Sets up the model pool with configurations for models, tokenizer, and
        default model loading parameters. Automatically processes model kwargs
        to handle special configurations like torch_dtype parsing.

        Args:
            models: Configuration dictionary specifying the seq2seq models to manage.
                Keys are model names, values can be model paths/names or detailed configs.
            tokenizer: Configuration for the tokenizer to use with the models.
                Can be a simple path/name or detailed configuration with _target_.
            model_kwargs: Default keyword arguments applied to all model loading
                operations. Common options include torch_dtype, device_map, etc.
                The torch_dtype field is automatically parsed from string to dtype.
            **kwargs: Additional arguments passed to the parent BaseModelPool.

        Example:
            ```python
            pool = Seq2SeqLMPool(
                models={
                    "base": "t5-base",
                    "large": {"_target_": "transformers.AutoModelForSeq2SeqLM",
                             "pretrained_model_name_or_path": "t5-large"}
                },
                tokenizer="t5-base",
                model_kwargs={"torch_dtype": "bfloat16"}
            )
            ```
        """
        super().__init__(models, **kwargs)
        self._tokenizer = tokenizer
        self._model_kwargs = model_kwargs
        if self._model_kwargs is None:
            self._model_kwargs = DictConfig({})
        with flag_override(self._model_kwargs, "allow_objects", True):
            if hasattr(self._model_kwargs, "torch_dtype"):
                self._model_kwargs.torch_dtype = parse_dtype(
                    self._model_kwargs.torch_dtype
                )

    def load_model(self, model_name_or_config: str | DictConfig, *args, **kwargs):
        """Load a sequence-to-sequence language model from the pool.

        Loads a seq2seq model using the parent class loading mechanism while
        automatically applying the pool's default model kwargs. The method
        merges the pool's model_kwargs with any additional kwargs provided,
        giving priority to the explicitly provided kwargs.

        Args:
            model_name_or_config: Either a string model name from the pool
                configuration or a DictConfig containing model loading parameters.
            *args: Additional positional arguments passed to the parent load_model method.
            **kwargs: Additional keyword arguments that override the pool's default
                model_kwargs. Common options include device, torch_dtype, etc.

        Returns:
            AutoModelForSeq2SeqLM: The loaded sequence-to-sequence language model.
        """
        model_kwargs = deepcopy(self._model_kwargs)
        model_kwargs.update(kwargs)
        return super().load_model(model_name_or_config, *args, **model_kwargs)

    def load_tokenizer(self, *args, **kwargs):
        """Load the tokenizer associated with the sequence-to-sequence models.

        Loads a tokenizer based on the tokenizer configuration provided during
        pool initialization. The tokenizer should be compatible with the seq2seq
        models in the pool and is typically used for preprocessing input text
        and postprocessing generated output.

        Args:
            *args: Additional positional arguments passed to the tokenizer constructor.
            **kwargs: Additional keyword arguments passed to the tokenizer constructor.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer instance compatible with
                the seq2seq models in this pool.

        Raises:
            AssertionError: If no tokenizer configuration is provided.
        """
        assert self._tokenizer is not None, "Tokenizer is not defined in the config"
        tokenizer = isinstance(self._tokenizer, *args, **kwargs)
        return tokenizer
