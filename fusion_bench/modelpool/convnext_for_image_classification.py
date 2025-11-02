import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    TypeVar,
    Union,
    override,
)

import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import nn

from fusion_bench import BaseModelPool, auto_register_config, get_rankzero_logger
from fusion_bench.tasks.clip_classification import get_classnames, get_num_classes

log = get_rankzero_logger(__name__)


def load_transformers_convnext(
    config_path: str, pretrained: bool, dataset_name: Optional[str]
):
    from transformers import AutoConfig, ConvNextForImageClassification

    if pretrained:
        model = ConvNextForImageClassification.from_pretrained(config_path)
    else:
        config = AutoConfig.from_pretrained(config_path)
        model = ConvNextForImageClassification(config)

    if dataset_name is None:
        return model

    classnames = get_classnames(dataset_name)
    id2label = {i: c for i, c in enumerate(classnames)}
    label2id = {c: i for i, c in enumerate(classnames)}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.num_labels = model.config.num_labels

    model.classifier = (
        nn.Linear(
            model.classifier.in_features,
            len(classnames),
            device=model.classifier.weight.device,
            dtype=model.classifier.weight.dtype,
        )
        if model.config.num_labels > 0
        else nn.Identity()
    )
    return model


@auto_register_config
class ConvNextForImageClassificationPool(BaseModelPool):
    def load_processor(self):
        from transformers import AutoImageProcessor

        if self.has_pretrained:
            config_path = self._models["_pretrained_"].config_path
        else:
            for model_cfg in self._models.values():
                if isinstance(model_cfg, str):
                    config_path = model_cfg
                    break
                if "config_path" in model_cfg:
                    config_path = model_cfg["config_path"]
                    break
        return AutoImageProcessor.from_pretrained(config_path)

    @override
    def load_model(self, model_name_or_config: Union[str, DictConfig], *args, **kwargs):
        log.debug(f"Loading model: {model_name_or_config}", stacklevel=2)
        if (
            isinstance(model_name_or_config, str)
            and model_name_or_config in self._models
        ):
            model_name_or_config = self._models[model_name_or_config]

        match model_name_or_config:
            case str() as model_path:
                from transformers import AutoModelForImageClassification

                model = AutoModelForImageClassification.from_pretrained(model_path)
            case dict() | DictConfig() as model_config:
                model = load_transformers_convnext(
                    model_config["config_path"],
                    pretrained=model_config.get("pretrained", True),
                    dataset_name=model_config.get("dataset_name", None),
                )
            case _:
                raise ValueError(
                    f"Unsupported model_name_or_config type: {type(model_name_or_config)}"
                )

        # override forward to return logits only
        original_forward = model.forward
        model.forward = lambda pixel_values, **kwargs: original_forward(
            pixel_values=pixel_values, **kwargs
        ).logits
        model.original_forward = original_forward

        return model

    @override
    def save_model(
        self,
        model,
        path,
        algorithm_config: Optional[DictConfig] = None,
        description: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model.save_pretrained(path)
        self.load_processor().save_pretrained(path)

        if algorithm_config is not None and rank_zero_only.rank == 0:
            from fusion_bench.models.hf_utils import create_default_model_card

            model_card_str = create_default_model_card(
                algorithm_config=algorithm_config,
                description=description,
                modelpool_config=self.config,
            )
            with open(os.path.join(path, "README.md"), "w") as f:
                f.write(model_card_str)
