import logging

from omegaconf import DictConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from fusion_bench.modelpool import ModelPool
from fusion_bench.utils import timeit_context

log = logging.getLogger(__name__)


class HuggingFaceGPT2ClassificationPool(ModelPool):
    def __init__(self, modelpool_config: DictConfig):
        super().__init__(modelpool_config)

        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if "_pretrained_" in self._model_names:
                self._tokenizer = GPT2Tokenizer.from_pretrained(
                    self.get_model_config("_pretrained_")["path"]
                )
            else:
                log.warning(
                    "No pretrained model found in the model pool. Returning the first model."
                )
                self._tokenizer = GPT2Tokenizer.from_pretrained(
                    self.get_model_config(self.model_names[0])["path"]
                )

    def load_model(
        self, model_config: str | DictConfig
    ) -> GPT2ForSequenceClassification:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
            f"Loading GPT2 classification model: '{model_config.name}' from '{model_config.path}'."
        ):
            model = GPT2ForSequenceClassification.from_pretrained(model_config.path)
        return model
