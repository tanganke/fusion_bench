import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib
import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm

TEST_DATASET = [
    "sun397",
    "stanford-cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
]
TEST_DATASET_LABELS = [
    "SUN397",
    "Cars",
    "RESISC45",
    "EuroSAT",
    "SVHN",
    "GTSRB",
    "MNIST",
    "DTD",
]

sns.set_theme(style="darkgrid")
color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# sns.set_palette("bright")

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

# load data from data directory

directory = Path("/home/enneng/fusion_bench-v0.2.4/fusion_bench/outputs/rankone_wemoe/clip-vit-base-patch32/layer_wise_routing_weights/")
num_layers = 12
num_experts = 128
target_layer = [1]

all_data = []
for task in tqdm(TEST_DATASET):
    # for layer_idx in range(num_layers):
    for layer_idx in target_layer:
        data = torch.load(
            directory / task / f"layer_{layer_idx}.pt",
            map_location="cpu",
            weights_only=True,
        )  # num_samples, num_token, num_experts
        data = data.mean(dim=1)  # num_samples, num_experts
        data = data.mean(dim=0)  # num_experts
        # print(data)
        all_data.append(data.tolist())
print(all_data)
