from omegaconf import DictConfig

from .base_pool import DictModelPool, ListModelPool, ModelPool, to_modelpool
from .huggingface_clip_vision import HuggingFaceClipVisionPool
from .huggingface_gpt2_classification import HuggingFaceGPT2ClassificationPool


def load_modelpool_from_config(modelpool_config: DictConfig):
    if hasattr(modelpool_config, "type"):
        if modelpool_config.type == "huggingface_clip_vision":
            return HuggingFaceClipVisionPool(modelpool_config)
        elif modelpool_config.type == "HF_GPT2ForSequenceClassification":
            return HuggingFaceGPT2ClassificationPool(modelpool_config)
        else:
            raise ValueError(f"Unknown model pool type: {modelpool_config.type}")
    else:
        raise ValueError("Model pool type not specified")
