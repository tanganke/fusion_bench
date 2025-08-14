import importlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["LIBRARY_PATH", "PROJECT_ROOT_PATH", "DEFAULT_CONFIG_PATH"]

LIBRARY_PATH = Path(importlib.import_module("fusion_bench").__path__[0])
"""Path to the library directory."""

PROJECT_ROOT_PATH = LIBRARY_PATH.parent
"""Path to the project root directory."""

if (PROJECT_ROOT_PATH / "config").is_dir():
    DEFAULT_CONFIG_PATH = PROJECT_ROOT_PATH / "config"
    """Path to the default config directory."""
elif (PROJECT_ROOT_PATH / "fusion_bench_config").is_dir():
    DEFAULT_CONFIG_PATH = PROJECT_ROOT_PATH / "fusion_bench_config"
else:
    log.warning("No default config path found.")
    DEFAULT_CONFIG_PATH = None
