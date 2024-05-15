import logging
from functools import cached_property

from omegaconf import DictConfig
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench.utils import timeit_context

from .base_pool import ModelPool

log = logging.getLogger(__name__)


class HuggingFaceClipVisionPool(ModelPool):
    """
    A model pool for managing Hugging Face's CLIP Vision models.

    This class extends the base `ModelPool` class and overrides its methods to handle
    the specifics of the CLIP Vision models provided by the Hugging Face Transformers library.
    """

    def __init__(self, modelpool_config: DictConfig):
        super().__init__(modelpool_config)

        self._clip_processor = None

    def clip_processor(self):
        if self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self.config["models"])
        return self._clip_processor

    def load_model(self, model_config: str | DictConfig) -> CLIPVisionModel:
        """
        Load a CLIP Vision model from the given configuration.

        Args:
            model_config (str | DictConfig): The configuration for the model to load.

        Returns:
            CLIPVisionModel: The loaded CLIP Vision model.
        """
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
            f"Loading CLIP vision model: '{model_config.name}' from '{model_config.path}'."
        ):
            vision_model = CLIPVisionModel.from_pretrained(model_config.path)
        return vision_model
