"""ResNet Model Pool for Image Classification.

This module provides a flexible model pool implementation for ResNet models used in image
classification tasks. It supports both torchvision and transformers implementations of ResNet
architectures with configurable preprocessing, loading, and saving capabilities.

Example Usage:
    Create a pool with a torchvision ResNet model:

    ```python
    >>> # Torchvision ResNet pool
    >>> pool = ResNetForImageClassificationPool(
    ...     type="torchvision",
    ...     models={"resnet18_cifar10": {"model_name": "resnet18", "dataset_name": "cifar10"}}
    ... )
    >>> model = pool.load_model("resnet18_cifar10")
    >>> processor = pool.load_processor(stage="train")
    ```

    Create a pool with a transformers ResNet model:

    ```python
    >>> # Transformers ResNet pool
    >>> pool = ResNetForImageClassificationPool(
    ...     type="transformers",
    ...     models={"resnet_model": {"config_path": "microsoft/resnet-50", "pretrained": True}}
    ... )
    ```
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

if TYPE_CHECKING:
    from torchvision.models import ResNet as TorchVisionResNet

log = get_rankzero_logger(__name__)


def load_torchvision_resnet(
    model_name: str, weights: Optional[str], num_classes: Optional[int]
) -> "TorchVisionResNet":
    """Load a ResNet model from torchvision with optional custom classifier head.

    This function creates a ResNet model using torchvision's model zoo and optionally
    replaces the final classification layer to match the required number of classes.

    Args:
        model_name (str): Name of the ResNet model to load (e.g., 'resnet18', 'resnet50').
            Must be a valid torchvision model name.
        weights (Optional[str]): Pretrained weights to load. Can be 'DEFAULT', 'IMAGENET1K_V1',
            or None for random initialization. See torchvision documentation for available options.
        num_classes (Optional[int]): Number of output classes. If provided, replaces the final
            fully connected layer. If None, keeps the original classifier (typically 1000 classes).

    Returns:
        TorchVisionResNet: The loaded ResNet model with appropriate classifier head.

    Raises:
        AttributeError: If model_name is not a valid torchvision model.

    Example:
        ```python
        >>> model = load_torchvision_resnet("resnet18", "DEFAULT", 10)  # CIFAR-10
        >>> model = load_torchvision_resnet("resnet50", None, 100)     # Random init, 100 classes
        ```
    """
    import torchvision.models

    model_fn = getattr(torchvision.models, model_name)
    model: "TorchVisionResNet" = model_fn(weights=weights)

    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_transformers_resnet(
    config_path: str, pretrained: bool, dataset_name: Optional[str]
):
    """Load a ResNet model from transformers with optional dataset-specific adaptation.

    This function creates a ResNet model using the transformers library and optionally
    adapts it for a specific dataset by updating the classifier head and label mappings.

    Args:
        config_path (str): Path or identifier for the model configuration. Can be a local path
            or a Hugging Face model identifier (e.g., 'microsoft/resnet-50').
        pretrained (bool): Whether to load pretrained weights. If True, loads from the
            specified config_path. If False, initializes with random weights using the config.
        dataset_name (Optional[str]): Name of the target dataset for adaptation. If provided,
            updates the model's classifier and label mappings to match the dataset's classes.
            If None, keeps the original model configuration.

    Returns:
        ResNetForImageClassification: The loaded and optionally adapted ResNet model.

    Example:
        ```python
        >>> # Load pretrained model adapted for CIFAR-10
        >>> model = load_transformers_resnet("microsoft/resnet-50", True, "cifar10")
        >>> # Load random initialized model with default classes
        >>> model = load_transformers_resnet("microsoft/resnet-50", False, None)
        ```
    """
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
    model.num_labels = model.config.num_labels

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
    """Model pool for ResNet-based image classification models.

    This class provides a unified interface for managing ResNet models from different sources
    (torchvision and transformers) with automatic preprocessing, loading, and saving capabilities.
    It supports multiple ResNet architectures and can automatically adapt models to different
    datasets by adjusting the number of output classes.

    The pool supports two main types:
    - "torchvision": Uses torchvision's ResNet implementations with standard ImageNet preprocessing
    - "transformers": Uses Hugging Face transformers' ResNetForImageClassification with auto processors

    Args:
        type (str): Model source type, must be either "torchvision" or "transformers".
        **kwargs: Additional arguments passed to the base BaseModelPool class.

    Attributes:
        type (str): The model source type specified during initialization.

    Raises:
        AssertionError: If type is not "torchvision" or "transformers".

    Example:
        Create a pool with a torchvision ResNet model:

        ```python
        >>> # Torchvision-based pool
        >>> pool = ResNetForImageClassificationPool(
        ...     type="torchvision",
        ...     models={
        ...         "resnet18_cifar10": {
        ...             "model_name": "resnet18",
        ...             "weights": "DEFAULT",
        ...             "dataset_name": "cifar10"
        ...         }
        ...     }
        ... )
        ```
        ```

        Create a pool with a transformers ResNet model:

        ```python
        >>> # Transformers-based pool
        >>> pool = ResNetForImageClassificationPool(
        ...     type="transformers",
        ...     models={
        ...         "resnet_model": {
        ...             "config_path": "microsoft/resnet-50",
        ...             "pretrained": True,
        ...             "dataset_name": "imagenet"
        ...         }
        ...     }
        ... )
        ```
    """

    def __init__(self, models, type: str, **kwargs):
        super().__init__(models=models, **kwargs)
        assert type in [
            "torchvision",
            "transformers",
        ], "type must be either 'torchvision' or 'transformers'"

    def load_processor(
        self, stage: Literal["train", "val", "test"] = "test", *args, **kwargs
    ):
        """Load the appropriate image processor/transform for the specified training stage.

        Creates stage-specific image preprocessing pipelines optimized for the model type:

        For torchvision models:
        - Train stage: Includes data augmentation (random resize crop, horizontal flip)
        - Val/test stages: Standard preprocessing (resize, center crop) without augmentation
        - All stages: Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        For transformers models:
        - Uses AutoImageProcessor from the pretrained model configuration
        - Automatically handles model-specific preprocessing requirements

        Args:
            stage (Literal["train", "val", "test"]): The training stage determining preprocessing type.
                - "train": Applies data augmentation for training
                - "val"/"test": Uses standard preprocessing for evaluation
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Union[transforms.Compose, AutoImageProcessor]: The image processor/transform pipeline
            appropriate for the specified stage and model type.

        Raises:
            ValueError: If no valid config_path can be found for transformers models.

        Example:
            ```python
            >>> # Get training transforms for torchvision model
            >>> train_transform = pool.load_processor(stage="train")
            >>> # Get evaluation processor for transformers model
            >>> eval_processor = pool.load_processor(stage="test")
            ```
        """
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
        """Load a ResNet model based on the provided configuration or model name.

        This method supports flexible model loading from different sources and configurations:
        - Direct model names (e.g., "resnet18", "resnet50") for standard architectures
        - Model pool keys that map to configurations
        - Dictionary/DictConfig objects with detailed model specifications
        - Hugging Face model identifiers for transformers models

        For torchvision models, supports:
        - Standard ResNet architectures: resnet18, resnet34, resnet50, resnet101, resnet152
        - Custom configurations with model_name, weights, and num_classes specifications
        - Automatic dataset adaptation with class number inference

        For transformers models:
        - Loading from Hugging Face Hub or local paths
        - Pretrained or randomly initialized models
        - Automatic logits extraction by overriding forward method
        - Dataset-specific label mapping configuration

        Args:
            model_name_or_config (Union[str, DictConfig]): Model specification that can be:
                - A string model name (e.g., "resnet18") for standard architectures
                - A model pool key referencing a stored configuration
                - A dict/DictConfig with model parameters like:
                  * For torchvision: {"model_name": "resnet18", "weights": "DEFAULT", "num_classes": 10}
                  * For transformers: {"config_path": "microsoft/resnet-50", "pretrained": True, "dataset_name": "cifar10"}
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Union[TorchVisionResNet, ResNetForImageClassification]: The loaded ResNet model
            configured for the specified task. For transformers models, the forward method
            is modified to return logits directly instead of the full model output.

        Raises:
            ValueError: If model_name_or_config type is invalid or if model type is unknown.
            AssertionError: If num_classes from dataset doesn't match explicit num_classes specification.

        Example:
            ```python
            >>> # Load standard torchvision model
            >>> model = pool.load_model("resnet18")

            >>> # Load with custom configuration
            >>> config = {"model_name": "resnet50", "weights": "DEFAULT", "dataset_name": "cifar10"}
            >>> model = pool.load_model(config)

            >>> # Load transformers model
            >>> config = {"config_path": "microsoft/resnet-50", "pretrained": True}
            >>> model = pool.load_model(config)
            ```
        """
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
                        pretrained=model_config.get("pretrained", True),
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
        """Save a ResNet model to the specified path using the appropriate format.

        This method handles model saving based on the model pool type:
        - For torchvision models: Saves only the state_dict using torch.save()
        - For transformers models: Saves the complete model and processor using save_pretrained()

        The saving format ensures compatibility with the corresponding loading mechanisms
        and preserves all necessary components for model restoration.

        Args:
            model: The ResNet model to save. Should be compatible with the pool's model type.
            path (str): Destination path for saving the model. For torchvision models, this
                should be a file path (e.g., "model.pth"). For transformers models, this
                should be a directory path where model files will be stored.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Raises:
            ValueError: If the model type is unknown or unsupported.

        Note:
            For transformers models, both the model weights and the associated image processor
            are saved to ensure complete reproducibility of the preprocessing pipeline.

        Example:
            ```python
            >>> # Save torchvision model
            >>> pool.save_model(model, "checkpoints/resnet18_cifar10.pth")

            >>> # Save transformers model (saves to directory)
            >>> pool.save_model(model, "checkpoints/resnet50_model/")
            ```
        """
        if self.type == "torchvision":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)
        elif self.type == "transformers":
            model.save_pretrained(path)
            self.load_processor().save_pretrained(path)

            if algorithm_config is not None and rank_zero_only.rank == 0:
                from fusion_bench.models.hf_utils import create_default_model_card

                model_card_str = create_default_model_card(
                    base_model=base_model,
                    algorithm_config=algorithm_config,
                    description=description,
                    modelpool_config=self.config,
                )
                with open(os.path.join(path, "README.md"), "w") as f:
                    f.write(model_card_str)
        else:
            raise ValueError(f"Unknown model type: {self.type}")
