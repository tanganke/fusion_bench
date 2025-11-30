#!/usr/bin/env python3
"""
This is the CLI script that is executed when the user runs the `fusion_bench` command.
The script is responsible for parsing the command-line arguments, loading the configuration file, and running the fusion algorithm.
"""

import importlib
import importlib.resources
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from fusion_bench.constants import PROJECT_ROOT_PATH
from fusion_bench.programs import BaseHydraProgram
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)


def _get_default_config_path():
    for config_path_root in [os.getcwd(), PROJECT_ROOT_PATH]:
        for config_dir in ["config", "fusion_bench_config"]:
            config_path = os.path.join(config_path_root, config_dir)
            if os.path.exists(config_path) and os.path.isdir(config_path):
                return os.path.abspath(config_path)
    return None


@hydra.main(
    config_path=_get_default_config_path(),
    config_name="fabric_model_fusion",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the FusionBench command-line interface.

    This function serves as the primary entry point for the `fusion_bench` CLI command.
    It is decorated with Hydra's main decorator to handle configuration management,
    command-line argument parsing, and configuration file loading.

    The function performs the following operations:
    1. Resolves any interpolations in the configuration using OmegaConf
    2. Instantiates the appropriate program class based on the configuration
    3. Executes the program's run method to perform the fusion task

    Args:
        cfg (DictConfig): The Hydra configuration object containing all settings
            for the fusion task. This includes method configuration, model pool
            configuration, task pool configuration, and other runtime parameters.
            The configuration is automatically loaded by Hydra from the specified
            config files and command-line overrides.

    Returns:
        None: This function doesn't return a value but executes the fusion
            program which may save results, log outputs, or perform other
            side effects as configured.

    Example:
        This function is typically called automatically when running:
        ```bash
        fusion_bench method=... modelpool=... taskpool=...
        ```

        The Hydra decorator handles parsing these command-line arguments and
        loading the corresponding configuration files to populate the cfg parameter.
    """
    OmegaConf.resolve(cfg)
    program: BaseHydraProgram = instantiate(cfg)

    # Validate that instantiation succeeded and returned an object with 'run' method
    if not hasattr(program, "run") or not callable(getattr(program, "run")):
        err_msg = (
            f"Expected an object with a callable 'run' method, but got {type(program).__name__}. "
            "Ensure that the configuration specifies a concrete program class with '_target_'."
        )
        if "_target_" not in cfg:
            err_msg += "\nThe '_target_' field is missing from the root configuration."
        else:
            err_msg += f"\nFound '_target_': {cfg._target_}"
        err_msg += f"\n\nConfiguration content:\n{cfg}"
        raise TypeError(err_msg)

    program.run()


if __name__ == "__main__":
    main()
