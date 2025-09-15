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
from omegaconf import DictConfig
from torch import nn

from fusion_bench import BaseModelPool, auto_register_config, get_rankzero_logger
from fusion_bench.tasks.clip_classification import get_classnames, get_num_classes

if TYPE_CHECKING:
    from torchvision.models import ResNet as TorchVisionResNet

log = get_rankzero_logger(__name__)


def load_torchvision_resnet(
    model_name: str, weights: Optional[str], num_classes: Optional[int]
) -> "TorchVisionResNet":
    import torchvision.models

    model_fn = getattr(torchvision.models, model_name)
    model: "TorchVisionResNet" = model_fn(weights=weights)

    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_transformers_resnet(
    config_path: str, pretrained: bool, dataset_name: Optional[str]
):
    from transformers import AutoConfig, ResNetForImageClassification

    if pretrained:
        model = ResNetForImageClassification.from_pretrained(config_path)
    else:
        config = AutoConfig.from_pretrained(config_path)
        model = ResNetForImageClassification(config)

    if dataset_name is None:
        return model

    classnames = get_classnames(dataset_name)
    id2label = {i: c for i, c in enumerate(classnames)}
    label2id = {c: i for i, c in enumerate(classnames)}
    model.config.id2label = id2label
    model.config.label2id = label2id

    model.classifier[1] = (
        nn.Linear(
            model.classifier[1].in_features,
            len(classnames),
        )
        if model.config.num_labels > 0
        else nn.Identity()
    )
    return model


@auto_register_config
class ResNetForImageClassificationPool(BaseModelPool):
    def __init__(self, type: str, **kwargs):
        super().__init__(**kwargs)
        assert type in ["torchvision", "transformers"]

    def load_processor(
        self, stage: Literal["train", "val", "test"] = "test", *args, **kwargs
    ):
        if self.type == "torchvision":
            from torchvision import transforms

            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            if stage == "train":
                train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        to_tensor,
                        normalize,
                    ]
                )
                return train_transform
            else:
                val_transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        to_tensor,
                        normalize,
                    ]
                )
                return val_transform

        elif self.type == "transformers":
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

        if self.type == "torchvision":
            from torchvision.models import (
                resnet18,
                resnet34,
                resnet50,
                resnet101,
                resnet152,
            )

            match model_name_or_config:
                case "resnet18":
                    model = resnet18()
                case "resnet34":
                    model = resnet34()
                case "resnet50":
                    model = resnet50()
                case "resnet101":
                    model = resnet101()
                case "resnet152":
                    model = resnet152()
                case dict() | DictConfig() as model_config:
                    if "dataset_name" in model_config:
                        num_classes = get_num_classes(model_config["dataset_name"])
                        if "num_classes" in model_config:
                            assert (
                                num_classes == model_config["num_classes"]
                            ), f"num_classes mismatch: {num_classes} vs {model_config['num_classes']}"
                    elif "num_classes" in model_config:
                        num_classes = model_config["num_classes"]
                    else:
                        num_classes = None
                    model = load_torchvision_resnet(
                        model_name=model_config["model_name"],
                        weights=model_config.get("weights", None),
                        num_classes=num_classes,
                    )
                case _:
                    raise ValueError(
                        f"Invalid model_name_or_config type: {type(model_name_or_config)}"
                    )
        elif self.type == "transformers":
            match model_name_or_config:
                case str() as model_path:
                    from transformers import AutoModelForImageClassification

                    model = AutoModelForImageClassification.from_pretrained(model_path)
                case dict() | DictConfig() as model_config:

                    model = load_transformers_resnet(
                        config_path=model_config["config_path"],
                        pretrained=model_config.get("pretrained", False),
                        dataset_name=model_config.get("dataset_name", None),
                    )
                case _:
                    raise ValueError(
                        f"Invalid model_name_or_config type: {type(model_name_or_config)}"
                    )

            # override forward to return logits only
            original_forward = model.forward
            model.forward = lambda pixel_values, **kwargs: original_forward(
                pixel_values=pixel_values, **kwargs
            ).logits
            model.original_forward = original_forward
        else:
            raise ValueError(f"Unknown model type: {self.type}")
        return model

    @override
    def save_model(self, model, path, *args, **kwargs):
        if self.type == "torchvision":
            torch.save(model.state_dict(), path)
        elif self.type == "transformers":
            model.save_pretrained(path)
            self.load_processor().save_pretrained(path)
        else:
            raise ValueError(f"Unknown model type: {self.type}")
