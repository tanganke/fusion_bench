import logging

from omegaconf import DictConfig
from torch.nn.modules import Module
from transformers import AutoModel

from fusion_bench.compat.modelpool import ModelPool

log = logging.getLogger(__name__)


class AutoModelPool(ModelPool):
    def load_model(self, model_config: str | DictConfig) -> Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        else:
            model_config = model_config

        model = AutoModel.from_pretrained(model_config.path)
        return model
