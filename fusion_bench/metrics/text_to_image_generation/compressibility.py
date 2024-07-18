"""
In this script, two reward functions are defined:

- compressibility, in which the file size of the image after JPEG compression is minimized
- incompressibility, in which the same measure is maximized.
"""

import io
from typing import List

import numpy as np
import torch
from PIL import Image


def jpeg_incompressibility_scorer():
    """
    Function to calculate the incompressibility score of an image.
    The score is calculated based on the size of the image after JPEG compression.
    The larger the size, the higher the incompressibility score.
    """

    def _fn(images: torch.Tensor, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        images: List[Image.Image] = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return torch.asarray(sizes), {}

    return _fn


def jpeg_compressibility_scorer():
    """
    Function to calculate the compressibility score of an image.
    The score is calculated based on the size of the image after JPEG compression.
    The smaller the size, the higher the compressibility score.
    """
    jpeg_fn = jpeg_incompressibility_scorer()

    def _fn(images: torch.Tensor, prompts, metadata):
        reward, metadata = jpeg_fn(images, prompts, metadata)
        return -reward, metadata

    return _fn
