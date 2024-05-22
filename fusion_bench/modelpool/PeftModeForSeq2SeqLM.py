import logging

import peft
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from fusion_bench.utils import timeit_context

from .base_pool import ModelPool

log = logging.getLogger(__name__)


class PeftModelForSeq2SeqLMPool(ModelPool):

    def load_model(self, model_config: str | DictConfig):
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        with timeit_context(f"Loading model {model_config['name']}"):
            if model_config["name"] == "_pretrained_":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_config["path"])
                return model
            else:
                model = self.load_model("_pretrained_")
                peft_model = PeftModel.from_pretrained(
                    model,
                    model_config["path"],
                    is_trainable=model_config.get("is_trainable", True),
                )
                if model_config.get("merge_and_unload", True):
                    return peft_model.merge_and_unload()
                else:
                    return peft_model
