import pickle
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

from fusion_bench.utils.validation import ValidationError, validate_file_exists


class InfiniteDataLoader:
    """
    A wrapper class for DataLoader to create an infinite data loader.
    This is useful in case we are only interested in the number of steps and not the number of epochs.

    This class wraps a DataLoader and provides an iterator that resets
    when the end of the dataset is reached, creating an infinite loop.

    Attributes:
        data_loader (DataLoader): The DataLoader to wrap.
        _data_iter (iterator): An iterator over the DataLoader.
        _iteration_count (int): Number of complete iterations through the dataset.

    Example:
        >>> train_loader = DataLoader(dataset, batch_size=32)
        >>> infinite_loader = InfiniteDataLoader(train_loader)
        >>> for i, batch in enumerate(infinite_loader):
        ...     if i >= 1000:  # Train for 1000 steps
        ...         break
        ...     train_step(batch)
    """

    def __init__(self, data_loader: DataLoader, max_retries: int = 1):
        """
        Initialize the InfiniteDataLoader.

        Args:
            data_loader: The DataLoader to wrap.
            max_retries: Maximum number of retry attempts when resetting the data loader (default: 1).

        Raises:
            ValidationError: If data_loader is None or not a DataLoader instance.
        """
        if data_loader is None:
            raise ValidationError(
                "data_loader cannot be None", field="data_loader", value=data_loader
            )

        self.data_loader = data_loader
        self.max_retries = max_retries
        self._data_iter = iter(data_loader)
        self._iteration_count = 0

    def __iter__(self):
        """Reset the iterator to the beginning."""
        self._data_iter = iter(self.data_loader)
        self._iteration_count = 0
        return self

    def __next__(self):
        """
        Get the next batch, resetting to the beginning when the dataset is exhausted.

        Returns:
            The next batch from the data loader.

        Raises:
            RuntimeError: If the data loader consistently fails to produce data.
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                data = next(self._data_iter)
                return data
            except StopIteration:
                # Dataset exhausted or dataloader is empty, reset to beginning
                self._iteration_count += 1
                try:
                    self._data_iter = iter(self.data_loader)
                    data = next(self._data_iter)
                    return data
                except Exception as e:
                    last_exception = e
                    continue
            except Exception as e:
                # Handle other potential errors from the data loader
                raise RuntimeError(
                    f"Error retrieving data from data loader: [{type(e).__name__}]{e}"
                ) from e

        # If we get here, all attempts failed
        raise RuntimeError(
            f"Failed to retrieve data from data loader after {self.max_retries} attempts. "
            f"Last error: [{type(last_exception).__name__}]{last_exception}. "
            + (
                f"The data loader appears to be empty."
                if isinstance(last_exception, StopIteration)
                else ""
            )
        ) from last_exception

    def reset(self):
        """Manually reset the iterator to the beginning of the dataset."""
        self._data_iter = iter(self.data_loader)
        self._iteration_count = 0

    @property
    def iteration_count(self) -> int:
        """Get the number of complete iterations through the dataset."""
        return self._iteration_count

    def __len__(self) -> int:
        """
        Return the length of the underlying data loader.

        Returns:
            The number of batches in one complete iteration.
        """
        return len(self.data_loader)


def load_tensor_from_file(
    file_path: Union[str, Path], device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Loads a tensor from a file, which can be either a .pt, .pth or .np file.
    If the file is not one of these formats, it will try to load it as a pickle file.

    Args:
        file_path (str): The path to the file to load.
        device: The device to move the tensor to. By default the tensor is loaded on the CPU.

    Returns:
        torch.Tensor: The tensor loaded from the file.

    Raises:
        ValidationError: If the file doesn't exist
        ValueError: If the file format is unsupported
    """
    # Validate file exists
    validate_file_exists(file_path)

    if file_path.endswith(".np"):
        tensor = torch.from_numpy(np.load(file_path)).detach_()
    if file_path.endswith((".pt", ".pth")):
        tensor = torch.load(file_path, map_location="cpu").detach_()
    else:
        try:
            tensor = pickle.load(open(file_path, "rb"))
        except Exception:
            raise ValueError(f"Unsupported file format: {file_path}")

    # Move tensor to device
    assert isinstance(tensor, torch.Tensor), f"Expected tensor, got {type(tensor)}"
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def train_validation_split(
    dataset: Dataset,
    validation_fraction: Optional[float] = 0.1,
    validation_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    return_split: Literal["all", "train", "val"] = "both",
) -> Union[Tuple[Dataset, Dataset], Dataset]:
    """
    Split a dataset into a training and validation set.

    Args:
        dataset (Dataset): The dataset to split.
        validation_fraction (Optional[float]): The fraction of the dataset to use for validation.
        validation_size (Optional[int]): The number of samples to use for validation. `validation_fraction` must be set to `None` if this is provided.
        random_seed (Optional[int]): The random seed to use for reproducibility.
        return_split (Literal["all", "train", "val"]): The split to return.

    Returns:
        Tuple[Dataset, Dataset]: The training and validation datasets.
    """
    # Check the input arguments
    assert (
        validation_fraction is None or validation_size is None
    ), "Only one of validation_fraction and validation_size can be provided"
    assert (
        validation_fraction is not None or validation_size is not None
    ), "Either validation_fraction or validation_size must be provided"

    # Compute the number of samples for training and validation
    num_samples = len(dataset)
    if validation_size is None:
        assert (
            0 < validation_fraction < 1
        ), "Validation fraction must be between 0 and 1"
        num_validation_samples = int(num_samples * validation_fraction)
        num_training_samples = num_samples - num_validation_samples
    else:
        assert (
            validation_size < num_samples
        ), "Validation size must be less than num_samples"
        num_validation_samples = validation_size
        num_training_samples = num_samples - num_validation_samples

    # Split the dataset
    generator = (
        torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    )
    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [num_training_samples, num_validation_samples], generator=generator
    )

    # return the split as requested
    if return_split == "all":
        return training_dataset, validation_dataset
    elif return_split == "train":
        return training_dataset
    elif return_split == "val":
        return validation_dataset
    else:
        raise ValueError(f"Invalid return_split: {return_split}")


def train_validation_test_split(
    dataset: Dataset,
    validation_fraction: float,
    test_fraction: float,
    random_seed: Optional[int] = None,
    return_spilt: Literal["all", "train", "val", "test"] = "all",
) -> Union[Tuple[Dataset, Dataset, Dataset], Dataset]:
    """
    Split a dataset into a training, validation and test set.

    Args:
        dataset (Dataset): The dataset to split.
        validation_fraction (float): The fraction of the dataset to use for validation.
        test_fraction (float): The fraction of the dataset to use for test.
        random_seed (Optional[int]): The random seed to use for reproducibility.
        return_spilt (Literal["all", "train", "val", "test"]): The split to return.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validation and test datasets.
    """
    num_samples = len(dataset)
    assert 0 < validation_fraction < 1, "Validation fraction must be between 0 and 1"
    assert 0 < test_fraction < 1, "Test fraction must be between 0 and 1"
    generaotr = (
        torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    )

    num_validation_samples = int(num_samples * validation_fraction)
    num_test_samples = int(num_samples * test_fraction)
    num_training_samples = num_samples - num_validation_samples - num_test_samples
    training_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [num_training_samples, num_validation_samples, num_test_samples],
        generator=generaotr,
    )

    # return the split as requested
    if return_spilt == "all":
        return training_dataset, validation_dataset, test_dataset
    elif return_spilt == "train":
        return training_dataset
    elif return_spilt == "val":
        return validation_dataset
    elif return_spilt == "test":
        return test_dataset
    else:
        raise ValueError(f"Invalid return_split: {return_spilt}")
