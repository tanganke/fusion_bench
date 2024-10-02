import pickle
from typing import Literal, Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


class InfiniteDataLoader:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data


def load_tensor_from_file(file_path: str, device=None) -> torch.Tensor:
    """
    Loads a tensor from a file, which can be either a .pt, .pth or .np file.
    If the file is not one of these formats, it will try to load it as a pickle file.

    Args:
        file_path (str): The path to the file to load.
        device: The device to move the tensor to. By default the tensor is loaded on the CPU.

    Returns:
        torch.Tensor: The tensor loaded from the file.
    """
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
    validation_fraction: float = 0.1,
    random_seed: Optional[int] = None,
    return_split: Literal["all", "train", "val"] = "both",
):
    """
    Split a dataset into a training and validation set.

    Args:
        dataset (Dataset): The dataset to split.
        validation_fraction (float): The fraction of the dataset to use for validation.
        random_seed (Optional[int]): The random seed to use for reproducibility.
        return_split (Literal["all", "train", "val"]): The split to return.

    Returns:
        Tuple[Dataset, Dataset]: The training and validation datasets.
    """
    num_samples = len(dataset)
    assert 0 < validation_fraction < 1, "Validation fraction must be between 0 and 1"
    generator = (
        torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    )

    num_validation_samples = int(num_samples * validation_fraction)
    num_training_samples = num_samples - num_validation_samples
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


def train_validation_test_split(
    dataset: Dataset,
    validation_fraction: float,
    test_fraction: float,
    random_seed: Optional[int] = None,
    return_spilt: Literal["all", "train", "val", "test"] = "all",
):
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
