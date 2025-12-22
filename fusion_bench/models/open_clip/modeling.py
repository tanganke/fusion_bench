"""
OpenCLIP model wrappers used by FusionBench.

This module provides lightweight `torch.nn.Module` wrappers around OpenCLIP
components that are commonly used throughout FusionBench experiments:

- `ImageEncoder`: loads an OpenCLIP image encoder and exposes `encode_image`.
- `ClassificationHead`: a linear head optionally normalizing inputs.
- `ImageClassifier` / `MultiHeadImageClassifier`: convenience compositions.

Note:
    This module requires the optional dependency `open_clip_torch`.
"""

from fusion_bench.utils.packages import is_open_clip_available

if not is_open_clip_available():
    raise ImportError(
        "open_clip is not installed. Please install it with `pip install open_clip_torch`."
    )

from pathlib import Path
from typing import Callable, List

import open_clip
import torch
from torch import Tensor

from . import utils
from .variables_and_paths import CACHEDIR, MODELS, OPENCLIP_CACHEDIR


class ImageEncoder(torch.nn.Module):
    R"""
    OpenCLIP image encoder wrapper.

    This class loads an OpenCLIP model by name and exposes a forward pass that
    returns image embeddings via `model.encode_image`.

    Args:
        model_name: A model name supported by `open_clip`. FusionBench also
            supports suffixes:
            - ``"__pretrained__<tag>"`` to select a specific pretrained weights tag.
            - ``"__init__"`` to use random initialization.
        keep_lang: If False (default), removes the text encoder (when present)
            to reduce memory usage.

    Examples:

        load the image encoder for a given model name

        >>> from fusion_bench.models.open_clip import ImageEncoder
        >>> image_encoder = ImageEncoder(model_name="ViT-B-32")
    """

    def __init__(self, model_name: str, keep_lang: bool = False):
        super().__init__()
        assert (
            model_name in MODELS
        ), f"Invalid model name: {model_name}. Valid models are: {MODELS}"

        if "__pretrained__" in model_name:
            name, pretrained = model_name.split("__pretrained__")
        elif "__init__" in model_name:
            print("Using random initialization.")
            name, pretrained = model_name.split("__init__")[0], None
        else:
            name = model_name
            pretrained = "openai"
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=OPENCLIP_CACHEDIR
        )

        self.cache_dir = CACHEDIR

        # if `keep_lang` is False, remove the text encoder to save memory
        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of images into embedding vectors."""
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def save(self, filename: str) -> None:
        """Serialize this module to disk."""
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name: str, filename: str | Path):
        """Load a saved encoder state dict into a freshly constructed encoder."""
        print(f"Loading image encoder from {filename}")

        state_dict = torch.load(filename, map_location="cpu")

        model = cls(model_name)
        model.load_state_dict(state_dict)
        return model


class ClassificationHead(torch.nn.Linear):
    """A linear classification head with optional input normalization.

    Args:
        normalize: If True, L2-normalize inputs along the last dimension before
            applying the linear projection.
        weights: Weight matrix of shape (num_classes, feature_dim).
        biases: Optional bias vector of shape (num_classes,).
    """

    def __init__(
        self,
        normalize: bool,
        weights: Tensor,
        biases: Tensor = None,
    ):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs: Tensor):
        """Compute logits from input features."""
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def save(self, filename):
        """Serialize this head to disk."""
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename, save_state_dict=False)

    @classmethod
    def load(cls, filename):
        """Load a serialized `ClassificationHead` instance from disk."""
        # print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    train_preprocess: Callable
    val_preprocess: Callable

    """Convenience module combining an `ImageEncoder` and a `ClassificationHead`."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        classification_head: ClassificationHead,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        """Disable gradient computation for the classification head."""
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs: Tensor):
        """Run encoder then head and return logits."""
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        """Serialize this module to disk."""
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized `ImageClassifier` instance from disk."""
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    """Image encoder with multiple task-specific classification heads."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        classification_heads: List[ClassificationHead],
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        """Disable gradient computation for all heads."""
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        """Run encoder then the selected head and return logits."""
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        """Serialize this module to disk."""
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized `MultiHeadImageClassifier` instance from disk."""
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
