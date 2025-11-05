# flake8: noqa F401
import importlib.metadata

from .paths import *
from .runtime import RuntimeConstants

# fusionbench version
try:
    FUSION_BENCH_VERSION = importlib.metadata.version("fusion-bench")
except importlib.metadata.PackageNotFoundError:
    # Fallback when package is not installed (e.g., during development)
    FUSION_BENCH_VERSION = "0.0.0.dev"
