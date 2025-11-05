# -*- coding: utf-8 -*-
"""
Image Corruption Module for Robustness Testing.

This module provides various image corruption functions to test model robustness.
It implements common corruptions such as noise, blur, compression artifacts, and
weather effects. These corruptions are commonly used in benchmark datasets like
ImageNet-C and CIFAR-10-C.

The corruptions can be applied at different severity levels (1-5), where higher
levels indicate stronger corruption effects.

Example:
    ```python
    from PIL import Image
    from fusion_bench.dataset.image_corruption.make_corruption import gaussian_noise, motion_blur

    # Load an image
    img = Image.open("example.jpg")

    # Apply gaussian noise at severity level 3
    corrupted_img = gaussian_noise(img, severity=3)

    # Apply motion blur at severity level 2
    blurred_img = motion_blur(img, severity=2)
    ```
"""

import logging

logger = logging.getLogger(__name__)

import collections
import warnings
from io import BytesIO

import cv2  # pip install opencv-python
import numpy as np
import skimage as sk
import torch
import torchvision.transforms as trn
from PIL import Image
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian  # pip install scikit-image
from tqdm import tqdm

try:
    from wand.api import library as wandlibrary
    from wand.image import Image as WandImage
except ImportError as e:
    logger.error(
        "Failed to import wand."
        "Install it with `apt-get install libmagickwand-dev` and `pip install Wand`"
        "For more information, refer to the documentation https://docs.wand-py.org/"
    )
    raise e

# /////////////// Distortion Helpers ///////////////

warnings.simplefilter("ignore", UserWarning)


# /////////////// Distortions ///////////////
class MotionImage(WandImage):
    """
    Extended WandImage class with motion blur capability.

    This class wraps ImageMagick's motion blur functionality through the Wand library.
    """

    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        """
        Apply motion blur effect to the image.

        Args:
            radius: The radius of the Gaussian, in pixels, not counting the center pixel.
            sigma: The standard deviation of the Gaussian, in pixels.
            angle: Apply the effect along this angle in degrees.
        """
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def gaussian_noise(x, severity=1):
    """
    Apply Gaussian noise corruption to an image.

    Adds random Gaussian noise to the image, simulating sensor noise or
    environmental interference.

    Args:
        x: Input image as PIL Image or numpy array. If numpy array, should be in
           range [0, 255].
        severity: Corruption severity level from 1 (mild) to 5 (severe).

    Returns:
        numpy.ndarray: Corrupted image as numpy array in range [0, 255].
    """
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def impulse_noise(x, severity=1):
    """
    Apply impulse (salt-and-pepper) noise corruption to an image.

    Randomly replaces pixels with either maximum or minimum intensity values,
    simulating transmission errors or faulty pixels.

    Args:
        x: Input image as PIL Image or numpy array. If numpy array, should be in
           range [0, 255].
        severity: Corruption severity level from 1 (mild) to 5 (severe).

    Returns:
        numpy.ndarray: Corrupted image as numpy array in range [0, 255].
    """
    c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def motion_blur(x, severity=1):
    """
    Apply motion blur corruption to an image.

    Simulates camera shake or object motion during image capture by applying
    directional blur at a random angle.

    Args:
        x: Input PIL Image.
        severity: Corruption severity level from 1 (mild) to 5 (severe).
           Higher severity increases blur radius and sigma.

    Returns:
        numpy.ndarray: Corrupted image as numpy array in range [0, 255].
           Returns RGB image regardless of input format.
    """
    c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]

    output = BytesIO()
    x.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (32, 32):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def spatter(x, severity=1):
    """
    Apply spatter corruption to an image.

    Simulates liquid splatter effects (water or mud) on the image, creating
    realistic occlusions similar to raindrops or dirt on a camera lens.

    Args:
        x: Input image as PIL Image or numpy array. If numpy array, should be in
           range [0, 255].
        severity: Corruption severity level from 1 (mild) to 5 (severe).
           Levels 1-3 simulate water splatter, levels 4-5 simulate mud splatter.

    Returns:
        numpy.ndarray: Corrupted image as numpy array in range [0, 255].
    """
    c = [
        (0.62, 0.1, 0.7, 0.7, 0.5, 0),
        (0.65, 0.1, 0.8, 0.7, 0.5, 0),
        (0.65, 0.3, 1, 0.69, 0.5, 0),
        (0.65, 0.1, 0.7, 0.69, 0.6, 1),
        (0.65, 0.1, 0.5, 0.68, 0.6, 1),
    ][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate(
            (
                175 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate(
            (
                63 / 255.0 * np.ones_like(x[..., :1]),
                42 / 255.0 * np.ones_like(x[..., :1]),
                20 / 255.0 * np.ones_like(x[..., :1]),
            ),
            axis=2,
        )

        color *= m[..., np.newaxis]
        x *= 1 - m[..., np.newaxis]

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    """
    Apply contrast reduction corruption to an image.

    Reduces image contrast by blending pixels toward their mean values,
    simulating poor lighting conditions or low-quality image sensors.

    Args:
        x: Input image as PIL Image or numpy array. If numpy array, should be in
           range [0, 255].
        severity: Corruption severity level from 1 (mild) to 5 (severe).
           Higher severity results in lower contrast.

    Returns:
        numpy.ndarray: Corrupted image as numpy array in range [0, 255].
    """
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def jpeg_compression(x, severity=1):
    """
    Apply JPEG compression artifacts to an image.

    Simulates compression artifacts from lossy JPEG encoding at various
    quality levels, commonly seen in heavily compressed images.

    Args:
        x: Input PIL Image.
        severity: Corruption severity level from 1 (mild) to 5 (severe).
           Lower severity uses higher JPEG quality (less compression).

    Returns:
        PIL.Image: Corrupted image as PIL Image.
    """
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    """
    Apply pixelation corruption to an image.

    Reduces image resolution by downsampling and then upsampling,
    creating a blocky, pixelated appearance.

    Args:
        x: Input PIL Image with size (32, 32).
        severity: Corruption severity level from 1 (mild) to 5 (severe).
           Higher severity results in more pixelation.

    Returns:
        PIL.Image: Corrupted image as PIL Image with original size (32, 32).

    Note:
        This function is designed for 32x32 images (e.g., CIFAR-10).
    """
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]

    x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
    x = x.resize((32, 32), PILImage.BOX)

    return x


# /////////////// End Distortions ///////////////


distortion_methods = collections.OrderedDict()
"""
OrderedDict mapping corruption names to their corresponding functions.

Available corruptions:
    - "Gaussian Noise": Additive Gaussian noise
    - "Impulse Noise": Salt-and-pepper noise
    - "Motion Blur": Directional motion blur
    - "Contrast": Reduced contrast
    - "Pixelate": Resolution reduction
    - "JPEG": JPEG compression artifacts
    - "Spatter": Water or mud splatter effects

Example:
    ```python
    from PIL import Image
    from fusion_bench.dataset.image_corruption.make_corruption import distortion_methods

    img = Image.open("example.jpg")
    for name, corruption_fn in distortion_methods.items():
        corrupted = corruption_fn(img, severity=3)
        # Process corrupted image
    ```
"""
distortion_methods["Gaussian Noise"] = gaussian_noise
distortion_methods["Impulse Noise"] = impulse_noise
distortion_methods["Motion Blur"] = motion_blur
distortion_methods["Contrast"] = contrast
distortion_methods["Pixelate"] = pixelate
distortion_methods["JPEG"] = jpeg_compression
distortion_methods["Spatter"] = spatter
