import os
from copy import deepcopy

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from fusion_bench.utils.dtype import parse_dtype

__all__ = ["separate_save", "separate_load"]


def dir_is_empty(path: str) -> bool:
    return not os.path.exists(path) or len(os.listdir(path)) == 0


def separate_save(
    model: nn.Module,
    save_dir: str,
    dtype=None,
    in_place: bool = True,
    model_file="functional.bin",
    state_dict_file="state_dict.bin",
    use_safe_tensors: bool = True,
):
    """
    Save the model's architecture and state dictionary separately.

    Args:
        model (nn.Module): The PyTorch model to save.
        save_dir (str): The directory where the model and state dictionary will be saved.
        in_place (bool, optional): If True, the original model is modified. If False, a deepcopy of the model is used. Default is True.
        model_file (str, optional): The name of the file to save the model's architecture. Default is "functional.bin".
        state_dict_file (str, optional): The name of the file to save the model's state dictionary. Default is "state_dict.bin".
    """
    if os.path.exists(save_dir) and not dir_is_empty(save_dir):
        raise FileExistsError(f"Directory exists and is not empty. {save_dir}")

    if not in_place:
        model = deepcopy(model)
    state_dict = {}
    for name, param in model.state_dict().items():
        state_dict[name] = param.clone().detach().to(dtype=dtype).cpu()

    model = model.to_empty(device="meta")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model, os.path.join(save_dir, model_file))
    if not use_safe_tensors:
        torch.save(state_dict, os.path.join(save_dir, state_dict_file))
    else:
        save_file(state_dict, os.path.join(save_dir, state_dict_file))


def separate_load(
    load_dir: str,
    strict: bool = True,
    dtype: torch.dtype = None,
    device: torch.device = "cpu",
    model_file="functional.bin",
    state_dict_file="state_dict.bin",
    use_safe_tensors: bool = True,
):
    """
    Load the model's architecture and state dictionary separately.

    Args:
        load_dir (str): The directory from which the model and state dictionary will be loaded.
        strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict() function. Default is True.
        model_file (str, optional): The name of the file from which to load the model's architecture. Default is "functional.bin".
        state_dict_file (str, optional): The name of the file from which to load the model's state dictionary. Default is "state_dict.bin".

    Returns:
        nn.Module: The loaded PyTorch model with the state dictionary applied.
    """
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Directory {load_dir} does not exist.")
    dtype = parse_dtype(dtype)

    model: nn.Module = (
        torch.load(os.path.join(load_dir, model_file))
        .to(dtype=dtype)
        .to_empty(device=device or "cpu")
    )
    if state_dict_file is not None:
        if not use_safe_tensors:
            state_dict = torch.load(
                os.path.join(load_dir, state_dict_file),
                map_location="cpu",
            )
        else:
            state_dict = {}
            with safe_open(
                os.path.join(load_dir, state_dict_file), framework="pt", device="cpu"
            ) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        if dtype is not None:
            for name, param in state_dict.items():
                state_dict[name] = param.to(dtype=dtype, non_blocking=True)

        model.load_state_dict(state_dict, strict=strict)
    return model
