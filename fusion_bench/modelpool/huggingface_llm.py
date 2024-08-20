import functools
import logging
from typing import Optional

from omegaconf import DictConfig
from torch.nn.modules import Module
from transformers import AutoModel, AutoModelForCausalLM

from fusion_bench.modelpool.base_pool import ModelPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.dtype import parse_dtype

log = logging.getLogger(__name__)


class AutoModelForCausalLMPool(ModelPool):
    def load_model(self, model_config: str | DictConfig) -> Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        else:
            model_config = model_config

        kwargs = {}
        if self.config.get("dtype", None) is not None:
            kwargs["torch_dtype"] = parse_dtype(self.config.dtype)

        with timeit_context(f"loading model from {model_config.path}"):
            model = AutoModelForCausalLM.from_pretrained(model_config.path, **kwargs)
        return model
