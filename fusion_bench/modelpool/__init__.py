from omegaconf import DictConfig

from .base_pool import DictModelPool, ListModelPool, ModelPool, to_modelpool
from .huggingface_clip_vision import HuggingFaceClipVisionPool
from .huggingface_gpt2_classification import HuggingFaceGPT2ClassificationPool
from .huggingface_llm import AutoModelForCausalLMPool
from .PeftModelForSeq2SeqLM import PeftModelForSeq2SeqLMPool
from .AutoModelForSeq2SeqLM import AutoModelForSeq2SeqLMPool
from .openclip_modelpool import OpenCLIPModelPool


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
    if hasattr(modelpool_config, "type"):
        if modelpool_config.type == "huggingface_clip_vision":
            return HuggingFaceClipVisionPool(modelpool_config)
        elif modelpool_config.type == "OpenCLIPModelPool":
            return OpenCLIPModelPool(modelpool_config)
        elif modelpool_config.type == "HF_GPT2ForSequenceClassification":
            return HuggingFaceGPT2ClassificationPool(modelpool_config)
        elif modelpool_config.type == "AutoModelForCausalLMPool":
            return AutoModelForCausalLMPool(modelpool_config)
        elif modelpool_config.type == "AutoModelForSeq2SeqLMPool":
            return AutoModelForSeq2SeqLMPool(modelpool_config)
        elif modelpool_config.type == "PeftModelForSeq2SeqLMPool":
            return PeftModelForSeq2SeqLMPool(modelpool_config)
        else:
            raise ValueError(f"Unknown model pool type: {modelpool_config.type}")
    else:
        raise ValueError("Model pool type not specified")
