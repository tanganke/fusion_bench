"""
This module provides a class to convert a dataset whose object is a list of dictionaries with keys "image" and "label" to a dataset whose object is a tuple of tensors (inputs, label) for CLIP models.
"""

from typing import Optional, Tuple

import torch
from transformers import CLIPProcessor, ProcessorMixin

__all__ = ["CLIPDataset"]


class CLIPDataset(torch.utils.data.Dataset):
    """
    A dataset class for CLIP models that converts a dataset of dictionaries or tuples
    into a format suitable for CLIP processing.

    This class wraps an existing dataset and applies CLIP preprocessing to the images.
    It expects each item in the dataset to be either a dictionary with 'image' and 'label' keys,
    or a tuple/list of (image, label).

    Args:
        dataset: The original dataset to wrap.
        processor (CLIPProcessor): The CLIP processor for preparing inputs. If None, no preprocessing is applied and raw images are returned.

    Attributes:
        dataset: The wrapped dataset.
        processor (CLIPProcessor): The CLIP processor used for image preprocessing.
    """

    def __init__(self, dataset, processor: Optional[CLIPProcessor] = None):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves and processes an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the processed image tensor and the label.

        Raises:
            ValueError: If the item is neither a dictionary nor a tuple/list of length 2.
        """
        item = self.dataset[idx]
        if isinstance(item, dict):
            item = item
        elif isinstance(item, (tuple, list)):
            assert len(item) == 2, "Each item should be a tuple or list of length 2"
            item = {"image": item[0], "label": item[1]}
        else:
            raise ValueError("Each item should be a dictionary or a tuple of length 2")
        image = item["image"]
        if self.processor is not None:
            if isinstance(self.processor, ProcessorMixin):
                # Apply the processor to the image to get the input tensor
                inputs = self.processor(images=[image], return_tensors="pt")[
                    "pixel_values"
                ][0]
            elif callable(self.processor):
                inputs = self.processor(image)
            else:
                raise ValueError(
                    "The processor should be a CLIPProcessor or a callable function"
                )
        else:
            # if processor is None, return the raw image directly
            inputs = image
        # convert boolean label to int, this is for the case when the label is a binary classification task
        if isinstance(item["label"], bool):
            item["label"] = 1 if item["label"] else 0
        return inputs, item["label"]
