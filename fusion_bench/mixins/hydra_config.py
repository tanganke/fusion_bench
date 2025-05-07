import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import hydra.core.global_hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from fusion_bench.utils import import_object, instantiate
from fusion_bench.utils.instantiate import set_print_function_call

log = logging.getLogger(__name__)


class HydraConfigMixin:
    """
    A mixin for classes that need to be instantiated from a config file.
    """

    @classmethod
    def from_config(
        cls,
        config_name: Union[str, Path],
        overrides: Optional[List[str]] = None,
    ):
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            raise RuntimeError("Hydra is not initialized.")
        else:
            cfg = compose(config_name=config_name, overrides=overrides)

        config_groups = config_name.split("/")[:-1]
        for config_group in config_groups:
            cfg = cfg[config_group]

        if "_target_" in cfg:
            # if the config has a _target_ key, check if it is equal to the class name
            target_cls = import_object(cfg["_target_"])
            if target_cls != cls:
                log.warning(
                    f"The _target_ key in the config is {cfg['_target_']}, but the class name is {cls.__name__}."
                )
            with set_print_function_call(False):
                obj = instantiate(cfg)
        else:
            obj = cls(**cfg)

        return obj
