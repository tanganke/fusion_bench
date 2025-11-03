"""
Hugging Face DINOv2 image classification model pool.

This module provides a `BaseModelPool` implementation that loads and saves
DINOv2 models for image classification via `transformers`. It optionally
reconfigures the classification head to match a dataset's class names and
overrides `forward` to return logits only for simpler downstream usage.

See also: `fusion_bench.modelpool.convnext_for_image_classification` for a
parallel implementation for ConvNeXt-based classifiers.
"""

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


def load_transformers_dinov2(
    config_path: str, pretrained: bool, dataset_name: Optional[str]
):
    """Create a DINOv2 image classification model from a config or checkpoint.

    Args:
        config_path: A model identifier or local path understood by
            `transformers.AutoConfig/AutoModel` (e.g., "facebook/dinov2-base").
        pretrained: If True, load weights via `from_pretrained`; otherwise, build
            the model from config only.
        dataset_name: Optional dataset key used by FusionBench to derive class
            names via `get_classnames`. When provided, the model's id/label maps
            are updated and the classifier head is resized accordingly.

    Returns:
        Dinov2ForImageClassification: A `transformers.Dinov2ForImageClassification` instance. If
            `dataset_name` is set, the classifier head is adapted to the number of
            classes. The model's `config.id2label` and `config.label2id` are also
            populated.

    Notes:
        The overall structure mirrors the ConvNeXt implementation in
        `fusion_bench.modelpool.convnext_for_image_classification`.
    """
    from transformers import AutoConfig, Dinov2ForImageClassification

    if pretrained:
        model = Dinov2ForImageClassification.from_pretrained(config_path)
    else:
        config = AutoConfig.from_pretrained(config_path)
        model = Dinov2ForImageClassification(config)

    if dataset_name is None:
        return model

    classnames = get_classnames(dataset_name)
    id2label = {i: c for i, c in enumerate(classnames)}
    label2id = {c: i for i, c in enumerate(classnames)}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.num_labels = model.config.num_labels

    # If the model is configured with a positive number of labels, resize the
    # classifier to match the dataset classes; otherwise leave it as identity.
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
class Dinov2ForImageClassificationPool(BaseModelPool):
    """Model pool for DINOv2 image classification models (HF Transformers)."""

    def load_processor(self, *args, **kwargs):
        """Load the paired image processor for this model pool.

        Uses the configured model's identifier or config path to retrieve the
        appropriate `transformers.AutoImageProcessor` instance. If a pretrained
        model entry exists in the pool configuration, it is preferred to derive
        the processor to ensure tokenization/normalization parity.
        """
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
        """Load a DINOv2 model described by a name, path, or DictConfig.

        Accepts either a string (pretrained identifier or local path) or a
        config mapping with keys: `config_path`, optional `pretrained` (bool),
        and optional `dataset_name` to resize the classifier.

        Returns:
            A model whose `forward` is wrapped to return only logits to align
            with FusionBench expectations.
        """
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
                model = load_transformers_dinov2(
                    model_config["config_path"],
                    pretrained=model_config.get("pretrained", True),
                    dataset_name=model_config.get("dataset_name", None),
                )
            case _:
                raise ValueError(
                    f"Unsupported model_name_or_config type: {type(model_name_or_config)}"
                )

        # Override forward to return logits only, to unify the interface across
        # FusionBench model pools and simplify downstream usage.
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
        base_model: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Save the model, processor, and an optional model card to disk.

        Artifacts written to `path`:
        - The DINOv2 model via `model.save_pretrained`.
        - The paired image processor via `AutoImageProcessor.save_pretrained`.
        - If `algorithm_config` is provided and on rank-zero, a README model card
          documenting the FusionBench configuration.
        """
        model.save_pretrained(path)
        self.load_processor().save_pretrained(path)

        if algorithm_config is not None and rank_zero_only.rank == 0:
            from fusion_bench.models.hf_utils import create_default_model_card

            model_card_str = create_default_model_card(
                algorithm_config=algorithm_config,
                description=description,
                modelpool_config=self.config,
                base_model=base_model,
            )
            with open(os.path.join(path, "README.md"), "w") as f:
                f.write(model_card_str)
