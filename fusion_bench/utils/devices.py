import gc
import os
from typing import List, Optional, Union

import torch
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

__all__ = [
    "cuda_empty_cache",
    "to_device",
    "num_devices",
    "get_device",
    "get_current_device",
    "get_device_memory_info",
    "get_device_capabilities",
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


def get_current_device() -> torch.device:
    R"""
    Gets the current available device for PyTorch operations.
    This is used for distributed training.

    This function checks the availability of various types of devices in the following order:
    1. XPU (Intel's AI accelerator)
    2. NPU (Neural Processing Unit)
    3. MPS (Metal Performance Shaders, for Apple devices)
    4. CUDA (NVIDIA's GPU)
    5. CPU (Central Processing Unit, used as a fallback)

    The function returns the first available device found in the above order. If none of the specialized devices
    are available, it defaults to the CPU.

    Returns:
        torch.device: The current available device for PyTorch operations.

    Environment Variables:
        LOCAL_RANK: This environment variable is used to specify the device index for multi-device setups.
                    If not set, it defaults to "0".

    Example:
        >>> device = get_current_device()
        >>> print(device)
        xpu:0  # or npu:0, mps:0, cuda:0, cpu depending on availability
    """

    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_memory_info(device: torch.device, reset_stats: bool = True) -> dict:
    """
    Get memory information for a given device.

    Args:
        device (torch.device): The device for which to get memory information.

    Returns:
        dict: A dictionary containing memory information for the given device.
    """
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        peak_memory_active = torch.cuda.memory_stats(device).get(
            "active_bytes.all.peak", 0
        )
        peak_mem_alloc = torch.cuda.max_memory_allocated(device)
        peak_mem_reserved = torch.cuda.max_memory_reserved(device)

        if reset_stats:
            torch.cuda.reset_peak_memory_stats(device)

        return {
            "total_memory": total_memory,
            "reserved_memory": reserved_memory,
            "allocated_memory": allocated_memory,
            "peak_memory_active": peak_memory_active,
            "peak_memory_allocated": peak_mem_alloc,
            "peak_memory_reserved": peak_mem_reserved,
        }
    else:
        raise ValueError(
            f"Memory information not available for device type: {device.type}"
        )


def get_device_capabilities(device: torch.device) -> dict:
    """
    Get capabilities information for a given device.

    Args:
        device (torch.device): The device for which to get capabilities information.

    Returns:
        dict: A dictionary containing capabilities information for the given device.
    """
    if device.type == "cuda":
        return {
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
            "multi_processor_count": torch.cuda.get_device_properties(
                device
            ).multi_processor_count,
        }
    else:
        raise ValueError(
            f"Capabilities information not available for device type: {device.type}"
        )


def cleanup_cuda():
    """
    Call gc collect, empty CUDA cache, and reset peak memory stats.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def print_memory_usage(print_fn=print):
    """
    Print the current GPU memory usage.

    Returns:
        str: A string containing the allocated and cached memory in MB.
    """
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    cached = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    print_str = f"Allocated Memory: {allocated:.2f} MB\nCached Memory: {cached:.2f} MB"
    print_fn(print_str)
    return print_str
