import logging
from copy import deepcopy
from typing import List, Mapping, Union

import torch
from torch import Tensor, nn

from ..utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
)
from ..utils.type import _StateDict
from .base_algorithm import ModelFusionAlgorithm
from ..modelpool import ModelPool

log = logging.getLogger(__name__)


class EnsembleAlgorithm(ModelFusionAlgorithm):

    @torch.no_grad()
    def fuse(self, modelpool: ModelPool):
        pass
