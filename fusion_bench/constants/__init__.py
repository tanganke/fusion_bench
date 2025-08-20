# flake8: noqa F401
import importlib.metadata

from .paths import *
from .runtime import RuntimeConstants

# fusionbench version
FUSION_BENCH_VERSION = importlib.metadata.version("fusion-bench")
