import logging
from copy import deepcopy
from typing import Optional

from omegaconf import DictConfig, open_dict
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from typing_extensions import override

from fusion_bench.utils import instantiate, timeit_context

from ..base_pool import BaseModelPool

log = logging.getLogger(__name__)


class CLIPVisionModelPool(BaseModelPool):
    """
    A model pool for managing Hugging Face's CLIP Vision models.

    This class extends the base `ModelPool` class and overrides its methods to handle
    the specifics of the CLIP Vision models provided by the Hugging Face Transformers library.
    """

    _config_mapping = BaseModelPool._config_mapping | {"_processor": "processor"}

    def __init__(
        self,
        models: DictConfig,
        *,
        processor: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)

        self._processor = processor

    def load_processor(self, *args, **kwargs) -> CLIPProcessor:
        assert self._processor is not None, "Processor is not defined in the config"
        processor = instantiate(self._processor, *args, **kwargs)
        return processor

    def load_clip_model(self, model_name: str, *args, **kwargs) -> CLIPModel:
        model_config = self._models[model_name]
        assert isinstance(model_config, DictConfig), "Model config must be a DictConfig"
        model_config = deepcopy(model_config)
        with open_dict(model_config):
            model_config._target_ = "transformers.CLIPModel.from_pretrained"
        clip_model = instantiate(model_config, *args, **kwargs)
        return clip_model

    @override
    def save_model(self, model: CLIPVisionModel, path: str):
        """
        Save a CLIP Vision model to the given path.

        Args:
            model (CLIPVisionModel): The model to save.
            path (str): The path to save the model to.
        """
        with timeit_context(f'Saving clip vision model to "{path}"'):
            model.save_pretrained(path)
