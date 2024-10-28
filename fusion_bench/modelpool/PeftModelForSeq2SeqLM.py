import logging

from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

from fusion_bench.compat.modelpool.base_pool import ModelPool
from fusion_bench.utils import timeit_context

log = logging.getLogger(__name__)


class PeftModelForSeq2SeqLMPool(ModelPool):
    def load_model(self, model_config: str | DictConfig):
        """
        Load a model based on the provided configuration.

        The configuration options of `model_config` are:

        - name: The name of the model. If it is "_pretrained_", a pretrained Seq2Seq language model is returned.
        - path: The path where the model is stored.
        - is_trainable: A boolean indicating whether the model parameters should be trainable. Default is `True`.
        - merge_and_unload: A boolean indicating whether to merge and unload the PEFT model after loading. Default is `True`.


        Args:
            model_config (str | DictConfig): The configuration for the model. This can be either a string (name of the model) or a DictConfig object containing the model configuration.


        Returns:
            model: The loaded model. If the model name is "_pretrained_", it returns a pretrained Seq2Seq language model. Otherwise, it returns a PEFT model.
        """
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
