"""
This script downloads all datasets in the TALL-20 benchmark for CLIP Vision models.
"""

import os

from hydra import compose, initialize
from tqdm import tqdm

from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.utils import instantiate

# Load configuration using Hydra
with initialize(
    version_base=None,
    config_path=os.path.relpath(
        _get_default_config_path(), start=os.path.dirname(__file__)
    ),
):
    cfg = compose(
        config_name="fabric_model_fusion",
        overrides=[
            "method=dummy",
            "modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20.yaml",
            "taskpool=dummy",
        ],
    )
    modelpool: CLIPVisionModelPool = instantiate(cfg.modelpool)

for train_task in tqdm(modelpool.train_dataset_names):
    _ = modelpool.load_train_dataset(train_task)
