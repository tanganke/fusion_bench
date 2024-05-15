from functools import cached_property

from omegaconf import DictConfig
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from .base_pool import ModelPool


class HuggingFaceClipVisionPool(ModelPool):
    def __init__(self, modelpool_config: DictConfig):
        super().__init__(modelpool_config)

    def clip_processor(self):
        if self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self.config["models"])
        return self._clip_processor
