import logging
from copy import deepcopy
from typing import Optional

from omegaconf import DictConfig, flag_override
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils import parse_dtype

log = logging.getLogger(__name__)


def load_lora_model(
    base_model_path: str,
    peft_model_path: str,
    is_trainable: bool = True,
    merge_and_unload: bool = True,
):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        is_trainable=is_trainable,
    )
    if merge_and_unload:
        model = model.merge_and_unload()
    return model


class Seq2SeqLMPool(BaseModelPool):
    _config_mapping = BaseModelPool._config_mapping | {
        "_tokenizer": "tokenizer",
        "_model_kwargs": "model_kwargs",
    }

    def __init__(
        self,
        models: DictConfig,
        *,
        tokenizer: Optional[DictConfig],
        model_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        self._tokenizer = tokenizer
        self._model_kwargs = model_kwargs
        if self._model_kwargs is None:
            self._model_kwargs = DictConfig({})
        with flag_override(self._model_kwargs, "allow_objects", True):
            if hasattr(self._model_kwargs, "torch_dtype"):
                self._model_kwargs.torch_dtype = parse_dtype(
                    self._model_kwargs.torch_dtype
                )

    def load_model(self, model_name_or_config: str | DictConfig, *args, **kwargs):
        model_kwargs = deepcopy(self._model_kwargs)
        model_kwargs.update(kwargs)
        return super().load_model(model_name_or_config, *args, **model_kwargs)

    def load_tokenizer(self, *args, **kwargs):
        assert self._tokenizer is not None, "Tokenizer is not defined in the config"
        tokenizer = isinstance(self._tokenizer, *args, **kwargs)
        return tokenizer
