import functools
import logging
from typing import Optional

from omegaconf import DictConfig
from torch.nn.modules import Module
from transformers import AutoModel, AutoModelForCausalLM

from .base_pool import ModelPool

log = logging.getLogger(__name__)


class AutoModelForCausalLMPool(ModelPool):
    def load_model(self, model_config: str | DictConfig) -> Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        else:
            model_config = model_config

        model = AutoModelForCausalLM.from_pretrained(model_config.path)
        return model
