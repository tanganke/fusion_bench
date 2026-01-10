from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import BaseImageProcessor, ProcessorMixin


class ImageClassificationDataset(Dataset):
    """
    A dataset class for image classification models that converts a dataset of dictionaries or tuples
    into a format suitable for model processing.

    This class wraps an existing dataset and applies preprocessing to the images.
    It expects each item in the dataset to be either a dictionary with 'image' and 'label' keys,
    or a tuple/list of (image, label).
    """

    def __init__(
        self,
        dataset: Dataset,
        processor: Optional[Union["ProcessorMixin", "BaseImageProcessor"]] = None,
    ):
        """
        Args:
            dataset (Dataset): The original dataset to wrap.
            processor (Optional[Union[ProcessorMixin, BaseImageProcessor]]): The processor for preparing inputs.
                If None, no preprocessing is applied and raw images are returned.
        """
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
        # Standardize the item to a dictionary format
        # {"image": ..., "label": ...}
        item = self.dataset[idx]
        if isinstance(item, dict):
            item = item
        elif isinstance(item, (tuple, list)):
            assert len(item) == 2, "Each item should be a tuple or list of length 2"
            item = {"image": item[0], "label": item[1]}
        else:
            raise ValueError("Each item should be a dictionary or a tuple of length 2")

        # Process the image using the provided processor, if any
        image = item["image"]
        if self.processor is not None:
            if isinstance(self.processor, (ProcessorMixin, BaseImageProcessor)):
                # Apply the processor to the image to get the input tensor
                image = image.convert("RGB")  # ensure image is in RGB format
                inputs = self.processor(images=[image], return_tensors="pt")[
                    "pixel_values"
                ][0]
            elif callable(self.processor):
                inputs = self.processor(image)
            else:
                raise ValueError(
                    "The processor should be a transformers Processor or a callable function"
                )
        else:
            # if processor is None, return the raw image directly
            inputs = image
        # convert boolean label to int, this is for the case when the label is a binary classification task
        if isinstance(item["label"], bool):
            item["label"] = 1 if item["label"] else 0
        return inputs, item["label"]
