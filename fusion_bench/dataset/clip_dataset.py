"""
This module provides a class to convert a dataset whose object is a list of dictionaries with keys "image" and "label" to a dataset whose object is a tuple of tensors (inputs, label) for CLIP models.
"""

from fusion_bench.utils import DeprecationWarningMeta

from .image_dataset import ImageClassificationDataset

__all__ = ["CLIPDataset"]


class CLIPDataset(ImageClassificationDataset, metaclass=DeprecationWarningMeta):
    pass
