"""
Example Usage:

```bash
fusion_bench \
    method=adamerging \
        method.name=clip_layer_wise_adamerging \
        method.save_merging_weights=merging_weights.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    fabric_logger.root_dir=outputs/logs/ViT-B-32 \
    fabric_logger.name=clip_layer_wise_adamerging_adam
```
"""

import functools
import itertools
import logging
import os

import torch
from omegaconf import DictConfig, open_dict
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from fusion_bench.dataset import CLIPDataset, load_dataset_from_config
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.tasks.clip_classification.clip_mixin import CLIPClassificationMixin
from fusion_bench.utils import timeit_context

from .layer_wise_adamerging import LayerWiseAdaMergingAlgorithm

log = logging.getLogger(__name__)


class CLIPLayerWiseAdaMergingAlgorithm(
    CLIPClassificationMixin,
    LayerWiseAdaMergingAlgorithm,
):

    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()
