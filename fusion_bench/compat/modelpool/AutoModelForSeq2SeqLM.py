import logging

from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM

from fusion_bench.utils import timeit_context

from .base_pool import ModelPool

log = logging.getLogger(__name__)


class AutoModelForSeq2SeqLMPool(ModelPool):
    def load_model(self, model_config: str | DictConfig):
        """
        Load a Seq2Seq language model based on the provided configuration.

        The configuration options are:

        - name: The name of the model.
        - path: The path where the model is stored.

        Args:
            model_config (str | DictConfig): The configuration for the model. This can be either a string (name of the model) or a DictConfig object containing the model configuration.

        Returns:
            model: The loaded Seq2Seq language model.
        """

        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        with timeit_context(f"Loading model {model_config['name']}"):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_config["path"])
            return model
