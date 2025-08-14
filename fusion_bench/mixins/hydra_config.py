"""
Hydra Configuration Mixin for FusionBench.

This module provides a mixin class that enables easy instantiation of objects
from Hydra configuration files. It's designed to work seamlessly with the
FusionBench configuration system and supports dynamic object creation based
on YAML configuration files.

The mixin integrates with Hydra's configuration management system to provide
a clean interface for creating objects from structured configurations.
"""

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union

import hydra.core.global_hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from fusion_bench.utils import import_object, instantiate
from fusion_bench.utils.instantiate_utils import set_print_function_call

log = logging.getLogger(__name__)

T = TypeVar("T", bound="HydraConfigMixin")


class HydraConfigMixin:
    R"""
    A mixin class that provides configuration-based instantiation capabilities.

    This mixin enables classes to be instantiated directly from Hydra configuration
    files, supporting both direct instantiation and target-based instantiation patterns.
    It's particularly useful in FusionBench for creating model pools, task pools,
    and fusion algorithms from YAML configurations.

    The mixin handles:
    - Configuration loading and composition
    - Target class validation
    - Nested configuration group navigation
    - Object instantiation with proper error handling

    Example:

    ```python
    class MyAlgorithm(HydraConfigMixin):
        def __init__(self, param1: str, param2: int = 10):
            self.param1 = param1
            self.param2 = param2

    # Instantiate from config
    algorithm = MyAlgorithm.from_config("algorithms/my_algorithm")
    ```

    Note:
        This mixin requires Hydra to be properly initialized before use.
        Typically, this is handled by the main FusionBench CLI application.
    """

    @classmethod
    def from_config(
        cls,
        config_name: Union[str, Path],
        overrides: Optional[List[str]] = None,
    ) -> T:
        """
        Create an instance of the class from a Hydra configuration.

        This method loads a Hydra configuration file and instantiates the class
        using the configuration parameters. It supports both direct parameter
        passing and target-based instantiation patterns.

        Args:
            config_name: The name/path of the configuration file to load.
                        Can be a string like "algorithms/simple_average" or
                        a Path object. The .yaml extension is optional.
            overrides: Optional list of configuration overrides in the format
                      ["key=value", "nested.key=value"]. These allow runtime
                      modification of configuration parameters.

        Returns:
            An instance of the class configured according to the loaded configuration.

        Raises:
            RuntimeError: If Hydra is not properly initialized.
            ImportError: If a target class specified in the config cannot be imported.
            ValueError: If required configuration parameters are missing.

        Example:
            ```python
            # Load with basic config
            obj = MyClass.from_config("my_config")

            # Load with overrides
            obj = MyClass.from_config(
                "my_config",
                overrides=["param1=new_value", "param2=42"]
            )

            # Load nested config
            obj = MyClass.from_config("category/subcategory/my_config")
            ```

        Note:
            The method automatically handles nested configuration groups by
            navigating through the configuration hierarchy based on the
            config_name path structure.
        """
        # Verify Hydra initialization
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            raise RuntimeError(
                "Hydra is not initialized. Please ensure Hydra is properly "
                "initialized before calling from_config(). This is typically "
                "handled by the FusionBench CLI application."
            )
        else:
            # Compose the configuration with any provided overrides
            cfg = compose(config_name=config_name, overrides=overrides)

        # Navigate through nested configuration groups
        # E.g., "algorithms/simple_average" -> navigate to cfg.algorithms
        config_groups = config_name.split("/")[:-1]
        for config_group in config_groups:
            cfg = cfg[config_group]

        # Handle target-based instantiation
        if "_target_" in cfg:
            # Validate that the target class matches the calling class
            target_cls = import_object(cfg["_target_"])
            if target_cls != cls:
                log.warning(
                    f"Configuration target mismatch: config specifies "
                    f"'{cfg['_target_']}' but called on class '{cls.__name__}'. "
                    f"This may indicate a configuration error."
                )

            # Instantiate using the target pattern with function call logging disabled
            with set_print_function_call(False):
                obj = instantiate(cfg)
        else:
            # Direct instantiation using configuration as keyword arguments
            obj = cls(**cfg)

        return obj
