from typing import Callable, List

import open_clip
import torch
from torch import Tensor

from . import utils
from .variables_and_paths import CACHEDIR, MODELS, OPENCLIP_CACHEDIR


class ImageEncoder(torch.nn.Module):
    R"""
    Examples:

        load the image encoder for a given model name

        >>> from fusion_bench.models.open_clip import ImageEncoder
        >>> image_encoder = ImageEncoder(model_name="ViT-B-32")
    """

    def __init__(self, model_name: str, keep_lang=False):
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

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")

        state_dict = torch.load(filename, map_location="cpu")

        model = cls(model_name)
        model.load_state_dict(state_dict)
        return model


class ClassificationHead(torch.nn.Linear):
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
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename, save_state_dict=False)

    @classmethod
    def load(cls, filename):
        # print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    train_preprocess: Callable
    val_preprocess: Callable

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
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs: Tensor):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
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
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
