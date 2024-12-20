import torch
from torch import Tensor, nn

from fusion_bench import BaseAlgorithm

from .utils import TSVC_utils, check_parameterNamesMatch


class TaskSingularVectorCompression(BaseAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, modelpool):
        raise NotImplementedError(
            "Task Singular Vector Compression is not implemented yet."
        )
