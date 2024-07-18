import logging

from omegaconf import DictConfig
from torch import nn
from transformers import GPT2ForSequenceClassification, GPT2Model, GPT2Tokenizer

from fusion_bench.dataset.gpt2_glue import TokenizedGLUE
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
            log.info(f"Loading tokenizer classification model.")
            if "_pretrained_" in self._model_names:
                tokenizer = GPT2Tokenizer.from_pretrained(
                    self.get_model_config("_pretrained_")["path"]
                )
            else:
                log.warning(
                    "No pretrained model found in the model pool. Returning the first model."
                )
                tokenizer = GPT2Tokenizer.from_pretrained(
                    self.get_model_config(self.model_names[0])["path"]
                )
            tokenizer.model_max_length = 512
            if tokenizer.pad_token is None:
                if tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                elif tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    raise ValueError
            self._tokenizer = tokenizer
        return self._tokenizer

    def load_classifier(
        self, model_config: str | DictConfig
    ) -> GPT2ForSequenceClassification:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
            f"Loading GPT2 classification head from {model_config.path}."
        ):
            model = GPT2ForSequenceClassification.from_pretrained(model_config.path)
        return model

    def load_model(self, model_config: str | DictConfig) -> GPT2Model:
        model = self.load_classifier(model_config)
        return model.transformer

    def setup_taskpool(self, taskpool):
        if getattr(taskpool, "_tokenizer", None) is None:
            taskpool._tokenizer = self.tokenizer
        taskpool._modelpool = self

    def get_train_dataset(self, model_name: str):
        log.info('Loading train dataset: "{}"'.format(model_name))
        for dataset_config in self.config.train_datasets:
            if dataset_config.name == model_name:
                return TokenizedGLUE(tokenizer=self.tokenizer).load_dataset(
                    dataset_config.dataset.name
                )[dataset_config.dataset.split]
