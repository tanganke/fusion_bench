import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader


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
