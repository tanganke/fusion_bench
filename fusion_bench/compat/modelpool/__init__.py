# flake8: noqa F401
import warnings

from omegaconf import DictConfig

from fusion_bench.modelpool.huggingface_gpt2_classification import (
    GPT2ForSequenceClassificationPool,
)
from fusion_bench.modelpool.PeftModelForSeq2SeqLM import PeftModelForSeq2SeqLMPool

from .AutoModelForSeq2SeqLM import AutoModelForSeq2SeqLMPool
from .base_pool import DictModelPool, ListModelPool, ModelPool, to_modelpool
from .huggingface_clip_vision import HuggingFaceClipVisionPool


class ModelPoolFactory:
    """
    Factory class to create and manage different model pools.

    This class provides methods to create model pools based on a given configuration,
    register new model pools, and list available model pools.
    """

    _modelpool = {
        "NYUv2ModelPool": "fusion_bench.modelpool.nyuv2_modelpool.NYUv2ModelPool",
        "huggingface_clip_vision": HuggingFaceClipVisionPool,
        "HF_GPT2ForSequenceClassification": GPT2ForSequenceClassificationPool,
        "AutoModelPool": ".huggingface_automodel.AutoModelPool",
        # CausualLM
        "AutoModelForCausalLMPool": ".huggingface_llm.AutoModelForCausalLMPool",
        "LLamaForCausalLMPool": ".huggingface_llm.LLamaForCausalLMPool",
        "MistralForCausalLMPool": ".huggingface_llm.MistralForCausalLMPool",
        # Seq2SeqLM
        "AutoModelForSeq2SeqLMPool": AutoModelForSeq2SeqLMPool,
        "PeftModelForSeq2SeqLMPool": PeftModelForSeq2SeqLMPool,
    }

    @staticmethod
    def create_modelpool(modelpool_config: DictConfig) -> ModelPool:
        """
        Create an instance of a model pool based on the provided configuration.
        This is for v0.1.x versions, deprecated.
        For implementing new model pool, use `fusion_bench.modelpool.BaseModelPool` instead.

        Args:
            modelpool_config (DictConfig): The configuration for the model pool.
            Must contain a 'type' attribute that specifies the type of the model pool.

        Returns:
            ModelPool: An instance of the specified model pool.

        Raises:
            ValueError: If 'type' attribute is not found in the configuration or does not match any known model pool types.
        """
        warnings.warn(
            "ModelPoolFactory.create_modelpool() is deprecated and will be removed in future versions. "
            "Please implement new model pool using `fusion_bench.modelpool.BaseModelPool` instead.",
            DeprecationWarning,
        )

        from fusion_bench.utils import import_object

        modelpool_type = modelpool_config.get("type")
        if modelpool_type is None:
            raise ValueError("Model pool type not specified")

        if modelpool_type not in ModelPoolFactory._modelpool:
            raise ValueError(
                f"Unknown model pool: {modelpool_type}, available model pools: {ModelPoolFactory._modelpool.keys()}. You can register a new model pool using `ModelPoolFactory.register_modelpool()` method."
            )
        modelpool_cls = ModelPoolFactory._modelpool[modelpool_type]
        if isinstance(modelpool_cls, str):
            if modelpool_cls.startswith("."):
                modelpool_cls = f"fusion_bench.compat.modelpool.{modelpool_cls[1:]}"
            modelpool_cls = import_object(modelpool_cls)
        return modelpool_cls(modelpool_config)

    @staticmethod
    def register_modelpool(name: str, modelpool_cls):
        """
        Register a new model pool with the factory.

        Args:
            name (str): The name of the model pool.
            modelpool_cls: The class of the model pool to register.
        """
        ModelPoolFactory._modelpool[name] = modelpool_cls

    @classmethod
    def available_modelpools(cls):
        """
        Get a list of available model pools.

        Returns:
            list: A list of available model pool names.
        """
        return list(cls._modelpool.keys())


def load_modelpool_from_config(modelpool_config: DictConfig):
    """
    Loads a model pool based on the provided configuration.

    The function checks the 'type' attribute of the configuration and returns an instance of the corresponding model pool.
    If the 'type' attribute is not found or does not match any known model pool types, a ValueError is raised.

    Args:
        modelpool_config (DictConfig): The configuration for the model pool. Must contain a 'type' attribute that specifies the type of the model pool.

    Returns:
        An instance of the specified model pool.

    Raises:
        ValueError: If 'type' attribute is not found in the configuration or does not match any known model pool types.
    """
    return ModelPoolFactory.create_modelpool(modelpool_config)
