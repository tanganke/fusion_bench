R"""
RanDeS: Randomized Delta Superposition

Implementation of "RanDeS: Randomized Delta Superposition for Multi-Model Compression"
paper link: http://arxiv.org/abs/2505.11204

Modified from https://github.com/Zhou-Hangyu/randes
"""

from .base_algorithm import SuperposedAlgorithmBase
from .modelsoup import SuperposedModelSoupAlgorithm
from .task_arithmetic import (
    SuperposedTaskArithmeticAlgorithm,
    SuperposedTaskArithmeticLoRAAlgorithm,
)
