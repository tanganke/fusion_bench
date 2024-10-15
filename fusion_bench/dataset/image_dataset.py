from typing import Any, Callable, Tuple

from torch.utils.data import Dataset


class TransformedImageDataset(Dataset):
    """
    A dataset class for image classification tasks that applies a transform to images.

    This class wraps an existing dataset and applies a specified transform to the images.
    It expects each item in the dataset to be either a dictionary with 'image' and 'label' keys,
    or a tuple/list of (image, label).

    Args:
        dataset: The original dataset to wrap.
        transform (Callable): A function/transform to apply on the image.

    Attributes:
        dataset: The wrapped dataset.
        transform (Callable): The transform to be applied to the images.
    """

    def __init__(self, dataset, transform: Callable):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieves and processes an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the processed image and the label.

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
        inputs = self.transform(image)
        return inputs, item["label"]
