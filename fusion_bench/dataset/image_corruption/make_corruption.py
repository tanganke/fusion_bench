# -*- coding: utf-8 -*-
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
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]

    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def motion_blur(x, severity=1):
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
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]

    x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
    x = x.resize((32, 32), PILImage.BOX)

    return x


# /////////////// End Distortions ///////////////


distortion_methods = collections.OrderedDict()
distortion_methods["Gaussian Noise"] = gaussian_noise
distortion_methods["Impulse Noise"] = impulse_noise
distortion_methods["Motion Blur"] = motion_blur
distortion_methods["Contrast"] = contrast
distortion_methods["Pixelate"] = pixelate
distortion_methods["JPEG"] = jpeg_compression
distortion_methods["Spatter"] = spatter
