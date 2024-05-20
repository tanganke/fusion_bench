from omegaconf import DictConfig

from .base_pool import ListModelPool, ModelPool, to_modelpool
from .huggingface_clip_vision import HuggingFaceClipVisionPool


def load_modelpool_from_config(config: DictConfig):
    if hasattr(config, "type"):
        if config.type == "huggingface_clip_vision":
            return HuggingFaceClipVisionPool(config)
        else:
            raise ValueError(f"Unknown model pool type: {config.type}")
    else:
        raise ValueError("Model pool type not specified")
