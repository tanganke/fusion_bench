import gc
from typing import List, Optional, Union

import torch

__all__ = [
    "cuda_empty_cache",
    "to_device",
    "num_devices",
    "get_device",
]


def cuda_empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def to_device(obj, device: Optional[torch.device], **kwargs):
    """
    Move a given object to the specified device.

    This function recursively moves tensors, modules, lists, tuples, and dictionaries to the specified device.
    For unsupported types, the object is returned as is.

    Args:
        obj: The object to be moved to the device. This can be a torch.Tensor, torch.nn.Module, list, tuple, or dict.
        device (torch.device): The target device to move the object to. This can be `None`.
        **kwargs: Additional keyword arguments to be passed to the `to` method of torch.Tensor or torch.nn.Module. For example, `non_blocking=True`, `dtype=torch.float16`.

    Returns:
        The object moved to the specified device. The type of the returned object matches the type of the input object.

    Examples:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> to_device(tensor, torch.device('cuda'))
        tensor([1, 2, 3], device='cuda:0')

        >>> model = torch.nn.Linear(2, 2)
        >>> to_device(model, torch.device('cuda'))
        Linear(..., device='cuda:0')

        >>> data = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        >>> to_device(data, torch.device('cuda'))
        [tensor([1, 2], device='cuda:0'), tensor([3, 4], device='cuda:0')]
    """
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device, **kwargs)
    elif isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    elif isinstance(obj, dict):
        for key in obj:
            obj[key] = to_device(obj[key], device)
        return obj
    else:
        # the default behavior is to return the object as is
        return obj


def num_devices(devices: Union[int, List[int], str]) -> int:
    """
    Return the number of devices.

    Args:
        devices: `devices` can be a single int to specify the number of devices, or a list of device ids, e.g. [0, 1, 2, 3]， or a str of device ids, e.g. "0,1,2,3" and "[0, 1, 2]".

    Returns:
        The number of devices.
    """
    if isinstance(devices, int):
        return devices
    elif isinstance(devices, str):
        return len(devices.split(","))
    elif isinstance(devices, list):
        return len(devices)
    else:
        raise TypeError(
            f"devices must be a single int or a list of ints, but got {type(devices)}"
        )


def get_device(obj) -> torch.device:
    """
    Get the device of a given object.

    Args:
        obj: The object whose device is to be determined.

    Returns:
        torch.device: The device of the given object.

    Raises:
        ValueError: If the object type is not supported.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, torch.nn.Module):
        if hasattr(obj, "device"):
            return obj.device
        else:
            return next(iter(obj.parameters())).device
    elif isinstance(obj, torch.device):
        return obj
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
