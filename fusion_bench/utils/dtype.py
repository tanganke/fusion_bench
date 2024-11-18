import contextlib
from typing import Dict, Generator, Iterable, Optional, Tuple

import torch
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

PRECISION_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "fp32": torch.float32,
    "float32": torch.float32,
    "double": torch.float64,
    "fp64": torch.float64,
    "float64": torch.float64,
}


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
    if dtype not in PRECISION_STR_TO_DTYPE:
        raise ValueError(f"Unsupported dtype: {type(dtype)}")

    dtype = PRECISION_STR_TO_DTYPE[dtype]
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


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def infer_optim_dtype(model_dtype: "torch.dtype") -> "torch.dtype":
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    _is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
    try:
        _is_bf16_available = is_torch_bf16_gpu_available() or (
            is_torch_npu_available() and torch.npu.is_bf16_supported()
        )
    except Exception:
        _is_bf16_available = False

    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def validate_expected_param_dtype(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]], dtype: torch.dtype
) -> None:
    """
    Validates that all input parameters have the expected dtype.

    Args:
        named_params (Iterable[Tuple[str, torch.nn.Parameter]]): Iterable of named parameters.
        dtype (torch.dtype): Expected dtype.

    Raises:
        ValueError: If any parameter has a different dtype than `dtype`.
    """
    for name, param in named_params:
        if param.dtype != dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {dtype}"
            )
