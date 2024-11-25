import functools
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union, cast  # noqa: F401

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers.data import default_data_collator

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import GPT2ForSequenceClassificationPool
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import InfiniteDataLoader, load_tensor_from_file
from fusion_bench.utils.instantiate import instantiate

from .entropy_loss import entropy_loss
from .min_norm_solvers import MinNormSolver
from .utils import get_memory_usage

class FlanT5LayerWiseAdaMergingAlgorithm(BaseAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    