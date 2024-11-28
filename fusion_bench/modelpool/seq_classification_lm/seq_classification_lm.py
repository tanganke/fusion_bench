import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, Union, cast  # noqa: F401

from omegaconf import DictConfig, flag_override
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing_extensions import override

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils import instantiate
from fusion_bench.utils.dtype import parse_dtype

if TYPE_CHECKING:
    from transformers import LlamaForSequenceClassification

log = logging.getLogger(__name__)


class SeqenceClassificationModelPool(BaseModelPool):

    def __init__(
        self,
        models,
        *,
        tokenizer: Optional[DictConfig],
        model_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        # process `model_kwargs`
        self._tokenizer = tokenizer
        self._model_kwargs = model_kwargs
        if self._model_kwargs is None:
            self._model_kwargs = DictConfig({})
        with flag_override(self._model_kwargs, "allow_objects", True):
            if hasattr(self._model_kwargs, "torch_dtype"):
                self._model_kwargs.torch_dtype = parse_dtype(
                    self._model_kwargs.torch_dtype
                )

    @override
    def load_model(
        self,
        model_name_or_config: str | DictConfig,
        *args,
        **kwargs,
    ) -> Union[PreTrainedModel, "LlamaForSequenceClassification"]:
        model_kwargs = deepcopy(self._model_kwargs)
        model_kwargs.update(kwargs)
        if isinstance(model_name_or_config, str):
            log.info(f"Loading model: {model_name_or_config}", stacklevel=2)
        return super().load_model(model_name_or_config, *args, **model_kwargs)

    def load_tokenizer(self, *args, **kwargs) -> PreTrainedTokenizer:
        assert self._tokenizer is not None, "Tokenizer is not defined in the config"
        log.info("Loading tokenizer.", stacklevel=2)
        tokenizer = instantiate(self._tokenizer, *args, **kwargs)
        return tokenizer

    @override
    def save_model(
        self,
        model: PreTrainedModel,
        path: str,
        push_to_hub: bool = False,
        model_dtype: Optional[str] = None,
        save_tokenizer: bool = False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        """
        Save the model to the specified path.

        Args:
            model (PreTrainedModel): The model to be saved.
            path (str): The path where the model will be saved.
            push_to_hub (bool, optional): Whether to push the model to the Hugging Face Hub. Defaults to False.
            save_tokenizer (bool, optional): Whether to save the tokenizer along with the model. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `save_pretrained` method.
        """
        path = os.path.expanduser(path)
        if save_tokenizer:
            if tokenizer_kwargs is None:
                tokenizer_kwargs = {}
            # load the tokenizer
            tokenizer = self.load_tokenizer(**tokenizer_kwargs)
            tokenizer.save_pretrained(
                path,
                push_to_hub=push_to_hub,
            )
        if model_dtype is not None:
            model.to(dtype=parse_dtype(model_dtype))
        model.save_pretrained(
            path,
            push_to_hub=push_to_hub,
            **kwargs,
        )
