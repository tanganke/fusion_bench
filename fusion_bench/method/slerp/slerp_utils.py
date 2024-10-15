# Modification of: https://github.com/Digitous/LLM-SLERP-Merge/blob/main/slerpmergelm.py
# LLM HF SLERP Merge

# Retrofitted from dvschultz's script at https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
# to work with Huggingface Pretrained Language Models [by Chasm (AKA Digitous) and CalderaAI (on HuggingFace)].
# Original language model linear interpolation methods pioneered by Concedo AKA LostRuins on Github and HF.

# Idea for SLERP on LLMs sparked by discussion in Automatic1111 Stable Diffusion UI feature request for SLERP
# model merging for image diffusion domain models.

import logging
from typing import TypeVar

import numpy as np
import torch

log = logging.getLogger(__name__)
T = TypeVar("T", torch.Tensor, np.ndarray, float)


def lerp(t: float, v0: T, v1: T) -> T:
    """
    Performs linear interpolation between two tensors v0 and v1.

    Args:
        t (float): The interpolation factor, typically between 0 and 1.
        v0 (T): The starting value.
        v1 (T): The ending value.

    Returns:
        T: The interpolated value.
    """
    return (1 - t) * v0 + t * v1


def normalize(v: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Normalizes a tensor.

    Args:
        v (torch.Tensor): The tensor to normalize.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    norm = torch.linalg.norm(v)
    if norm > epsilon:
        return v / norm
    else:
        log.debug(f"Warning: Norm of v is very small ({norm}). Skipping normalization.")
        return v


def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    DOT_THRESHOLD=0.9995,
    epsilon=1e-8,
):
    """
    Performs spherical linear interpolation (slerp) between two tensors v0 and v1.

    Args:
        t (float): The interpolation factor, typically between 0 and 1.
        v0 (torch.Tensor): The starting tensor.
        v1 (torch.Tensor): The ending tensor.
        DOT_THRESHOLD (float, optional): Threshold for considering the vectors as collinear. Defaults to 0.9995.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: The interpolated tensor.
    """
    device = v0.device
    # Convert tensors to a common format, at least float32
    if v0.dtype != torch.float32 and v0.dtype != torch.float64:
        v0 = v0.to(dtype=torch.float32, non_blocking=True)
        v1 = v1.to(dtype=torch.float32, non_blocking=True)

    # Copy the vectors to reuse them later
    v0_copy = v0.clone()
    v1_copy = v1.clone()

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, epsilon)
    v1 = normalize(v1, epsilon)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = torch.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if torch.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
    else:
        # Calculate initial angle between v0 and v1
        theta_0 = torch.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        # Finish the slerp algorithm
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        res = s0 * v0_copy + s1 * v1_copy

    return res.to(device, non_blocking=True)
