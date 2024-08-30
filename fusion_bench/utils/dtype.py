from typing import Optional

import torch


def parse_dtype(dtype: Optional[str]):
    """
    Parses a string representation of a data type and returns the corresponding torch.dtype.

    Args:
        dtype (Optional[str]): The string representation of the data type.
                               Can be one of "float32", "float", "float64", "double",
                               "float16", "half", "bfloat16", or "bf16".
                               If None, returns None.

    Returns:
        torch.dtype: The corresponding torch.dtype if the input is a valid string representation.
                     If the input is already a torch.dtype, it is returned as is.
                     If the input is None, returns None.

    Raises:
        ValueError: If the input string does not correspond to a supported data type.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype is None:
        return None

    dtype = dtype.strip('"')
    if dtype == "float32" or dtype == "float":
        dtype = torch.float32
    elif dtype == "float64" or dtype == "double":
        dtype = torch.float64
    elif dtype == "float16" or dtype == "half":
        dtype = torch.float16
    elif dtype == "bfloat16" or dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {type(dtype)}")

    return dtype


def get_dtype(obj) -> torch.dtype:
    """
    Get the data type (dtype) of a given object.

    Returns:
        torch.dtype: The data type of the given object.

    Raises:
        ValueError: If the object type is not supported.
    """
    if isinstance(obj, torch.Tensor):
        return obj.dtype
    elif isinstance(obj, torch.nn.Module):
        if hasattr(obj, "dtype"):
            return obj.dtype
        else:
            return next(iter(obj.parameters())).dtype
    elif isinstance(obj, (torch.device, str)):
        return parse_dtype(obj)
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
