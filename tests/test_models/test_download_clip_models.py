import unittest

import torch
from torch import nn
from transformers import CLIPVisionModel


class TestDownloadCLIPModels(unittest.TestCase):
    def test_download_models(self):
        models = [
            "clip-vit-base-patch32",
            "clip-vit-base-patch16",
            "clip-vit-large-patch14",
        ]
        tasks = [
            "sun397",
            "stanford-cars",
            "resisc45",
            "eurosat",
            "svhn",
            "gtsrb",
            "mnist",
            "dtd",
            "oxford_flowers102",
            "pcam",
            "fer2013",
            "oxford-iiit-pet",
            "stl10",
            "cifar100",
            "cifar10",
            "fashion_mnist",
            "emnist_letters",
            "kmnist",
            "rendered-sst2",
        ]
        for model in models:
            for task in tasks:
                print(f"Downloading {model} for {task}")
                m = CLIPVisionModel.from_pretrained(f"tanganke/{model}_{task}")


if __name__ == "__main__":
    unittest.main()
