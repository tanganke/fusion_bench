import fnmatch
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NYUv2(Dataset):
    R"""
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:

        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]

    Modified from https://github.com/lorenmt/auto-lambda/blob/main/create_dataset.py

    removed the `augmentation` arg and add `transform` args
    """

    num_out_channels = {
        "segmentation": 13,
        "depth": 1,
        "normal": 3,
        "noise": 1,
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        sn_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
    ):
        """
        Initialize the NYUv2 dataset.

        Args:
            root (str): The root directory of the dataset.
            train (bool, optional): If True, use training set. If False, use validation set. Defaults to True.
            transform (Callable, optional): image transform. Defaults to None.
            seg_transform (Callable, optional): segmentation transform. Defaults to None.
            sn_transform (Callable, optional): surface normal transform. Defaults to None.
            depth_transform (Callable, optional): depth transform. Defaults to None.
        """
        self.root = os.path.expanduser(root)
        self.train = train

        self.transform = transform
        self.seg_transform = seg_transform
        self.sn_transform = sn_transform
        self.depth_transform = depth_transform

        if train:
            self.data_path = self.root + "/train"
        else:
            self.data_path = self.root + "/val"

        # calculate data length
        self.data_len = len(
            fnmatch.filter(os.listdir(self.data_path + "/image"), "*.npy")
        )
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and a dictionary of task-specific outputs.
        """
        # load data from the pre-processed npy files
        image = torch.from_numpy(
            np.moveaxis(
                np.load(self.data_path + "/image/{:d}.npy".format(index)), -1, 0
            )
        ).float()
        semantic = torch.from_numpy(
            np.load(self.data_path + "/label/{:d}.npy".format(index))
        ).float()
        depth = torch.from_numpy(
            np.moveaxis(
                np.load(self.data_path + "/depth/{:d}.npy".format(index)), -1, 0
            )
        ).float()
        normal = torch.from_numpy(
            np.moveaxis(
                np.load(self.data_path + "/normal/{:d}.npy".format(index)), -1, 0
            )
        ).float()
        noise = self.noise[index].float()

        if self.transform is not None:
            image = self.transform(image)
        if self.seg_transform is not None:
            semantic = self.seg_transform(semantic)
        if self.sn_transform is not None:
            normal = self.sn_transform(normal)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        return image, {
            "segmentation": semantic,
            "depth": depth,
            "normal": normal,
            "noise": noise,
        }

    def __len__(self):
        return self.data_len
