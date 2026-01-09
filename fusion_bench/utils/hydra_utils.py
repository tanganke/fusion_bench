import logging
import os

import hydra.core.hydra_config
from hydra import compose, initialize
from omegaconf import DictConfig

from fusion_bench.constants import PROJECT_ROOT_PATH

log = logging.getLogger(__name__)


def get_default_config_path():
    """
    Get the default configuration path by searching in common locations.
    """
    for config_path_root in [os.getcwd(), PROJECT_ROOT_PATH]:
        for config_dir in ["config", "fusion_bench_config"]:
            config_path = os.path.join(config_path_root, config_dir)
            if os.path.exists(config_path) and os.path.isdir(config_path):
                return os.path.abspath(config_path)
    return None


def initialize_hydra_config(
    config_name: str,
    overrides: list[str] = None,
    config_path: str = None,
    return_hydra_config: bool = False,
) -> DictConfig:
    """
    Load the Hydra configuration.

    Args:
        config_name (str): The name of the configuration file (without .yaml extension).
        overrides (list[str]): A list of configuration overrides.
        config_path (str): The path to the configuration directory. If None, it will be automatically detected.
        return_hydra_config (bool): If True, return the Hydra configuration object.

    Returns:
        DictConfig: The loaded configuration.

    Example:
        >>> cfg = initialize_hydra_config(
        ...     config_name="fabric_model_fusion",
        ...     overrides=["method=dummy", "modelpool=dummy"],
        ... )
        >>> print(cfg.method)
    """
    if config_path is None:
        config_path = get_default_config_path()

    # check config_path validity
    if config_path is None:
        raise FileNotFoundError("Could not find configuration directory.")
    if not os.path.isdir(config_path):
        raise NotADirectoryError(
            f"Configuration path {config_path} do not exists or is not a directory."
        )

    if overrides is None:
        overrides = []

    with initialize(
        version_base=None,
        config_path=os.path.relpath(
            config_path,
            start=os.path.dirname(__file__),
        ),
    ):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=return_hydra_config,
        )
        return cfg


def get_hydra_output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg.runtime.output_dir


def config_priority_get(priority_config, general_config, key, default):
    """
    Retrieve a configuration value with priority.

    This function retrieves the value associated with `key` from `priority_config` if it exists.
    If the key is not found in `priority_config`, it retrieves the value from `general_config`.
    If the key is not found in either configuration, it returns the provided `default` value.

    Args:
        priority_config (dict): The configuration dictionary with higher priority.
        general_config (dict): The general configuration dictionary.
        key (str): The key to look up in the configuration dictionaries.
        default: The default value to return if the key is not found in either configuration.

    Returns:
        The value associated with `key` from `priority_config` or `general_config`, or the `default` value if the key is not found.
    """
    if key in priority_config:
        return priority_config[key]
    return general_config.get(key, default)
