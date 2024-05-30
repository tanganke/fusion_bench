import logging
from typing import Optional

from omegaconf import DictConfig, open_dict

from fusion_bench.utils import timeit_context
from fusion_bench.models.openclip_model.src.modeling import ImageEncoder
from fusion_bench.models.openclip_model.clip_checkpoint_path import (
    pretrained_model_path,
    finetuned_model_path,
)
from .base_pool import ModelPool
import torch

log = logging.getLogger(__name__)


class OpenCLIPModelPool(ModelPool):
    def __init__(self, modelpool_config: DictConfig):
        from fusion_bench.models.openclip_model import setup_src

        super().__init__(modelpool_config)

        setup_src()

    def get_model_config(self, model_name: str) -> DictConfig:
        model_config = super().get_model_config(model_name)
        with open_dict(model_config):
            model_config.model_name = self.config.model_name
        return model_config

    def load_model(self, model_config: str | DictConfig) -> ImageEncoder:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        if model_config.name == "_pretrained_":
            model_path = pretrained_model_path(
                model_config.model_name,
                cache_dir=self.config.cache_dir,
            )
        else:
            model_path = finetuned_model_path(
                model_name=model_config.model_name,
                dataset_name=model_config.name,
                cache_dir=self.config.cache_dir,
            )
        with timeit_context(
            f"Loading OpenCLIP model: {model_config.name} from {model_path}."
        ):
            image_encoder = torch.load(model_path)
        return image_encoder
