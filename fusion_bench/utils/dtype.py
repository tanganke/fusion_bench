import torch


def parse_dtype(dtype: str):
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype is None:
        return None

    dtype = dtype.strip('"')
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64":
        dtype = torch.float64
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return dtype
