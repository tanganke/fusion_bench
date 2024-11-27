import functools
import logging
from typing import Optional

from omegaconf import DictConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from fusion_bench.dataset.gpt2_glue import TokenizedGLUE
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)
tokenizer: GPT2Tokenizer = None


@functools.cache
def load_gpt2_dataset(name: str, split: Optional[str] = None):
    global tokenizer
    dataset = TokenizedGLUE(tokenizer=tokenizer).load_dataset(name)
    if split is not None:
        dataset = dataset[split]
    return dataset


def load_gpt2_tokenizer(pretrained_model_name_or_path: str):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError
    return tokenizer


class GPT2ForSequenceClassificationPool(BaseModelPool):
    _config_mapping = BaseModelPool._config_mapping | {"_tokenizer": "tokenizer"}

    def __init__(self, tokenizer: DictConfig, **kwargs):
        self._tokenizer = tokenizer
        super().__init__(**kwargs)
        self.setup()

    def setup(self):
        global tokenizer
        self.tokenizer = tokenizer = instantiate(self._tokenizer)

    def load_classifier(
        self, model_config: str | DictConfig
    ) -> GPT2ForSequenceClassification:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config, return_copy=True)
        model_config._target_ = (
            "transformers.GPT2ForSequenceClassification.from_pretrained"
        )
        model = instantiate(model_config)
        return model


# For compatibility
HuggingFaceGPT2ClassificationPool = GPT2ForSequenceClassificationPool
