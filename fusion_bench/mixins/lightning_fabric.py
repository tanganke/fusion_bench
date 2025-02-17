import functools
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar

import lightning as L
import torch
from lightning.fabric.connector import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from fusion_bench.utils import import_object
from fusion_bench.utils.instantiate import instantiate

if TYPE_CHECKING:
    import lightning.fabric.loggers.tensorboard
    from lightning.fabric.strategies import FSDPStrategy

log = logging.getLogger(__name__)

TensorOrModule = TypeVar("TensorOrModule", torch.Tensor, torch.nn.Module, Any)


def get_policy(*args: str) -> set:
    """
    Get the policy from the provided list of policy names.

    Args:
        *args (str): A list of policy names.

    Returns:
        set: A set of policy objects.
    """
    return {import_object(arg) for arg in args}


def get_size_based_auto_wrap_policy(*args, **kwargs):
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    policy = functools.partial(size_based_auto_wrap_policy, *args, **kwargs)
    return policy


class LightningFabricMixin:
    """
    A mixin class for integrating Lightning Fabric into a project.

    This class provides methods to initialize and manage a Lightning Fabric instance for distributed computing,
    including setup with optional logging, device management for tensors and modules, and hyperparameter logging.
    It leverages the Lightning framework to facilitate distributed training and inference across multiple devices
    and nodes, with support for custom logging via TensorBoard.

    Attributes:
    - _fabric (L.Fabric): The Lightning Fabric instance used for distributed computing.

    Note:
    This mixin is designed to be used with classes that require distributed computing capabilities and wish to
    leverage the Lightning Fabric for this purpose. It assumes the presence of a `config` attribute or parameter
    in the consuming class for configuration.
    """

    _fabric_instance: L.Fabric = None

    def setup_lightning_fabric(self, config: DictConfig):
        """
        Initializes and launches the Lightning Fabric with optional logging.

        This method sets up the Lightning Fabric for distributed computing based on the provided configuration. If a fabric
        configuration is not found, it logs a warning and exits. Optionally, if a fabric logger configuration is provided,
        it initializes a TensorBoardLogger with the specified settings.

        Expected configuration keys:
        - fabric: The configuration for the Lightning Fabric.
        - fabric.loggers: The configuration for the TensorBoardLogger.
        """
        if self._fabric_instance is None:
            if config.get("fabric", None) is None:
                log.warning("No fabric configuration found. use default settings.")
                self._fabric_instance = L.Fabric()
            else:
                self._fabric_instance = instantiate(config.fabric)
            if not _is_using_cli():  # if not using cli, launch the fabric
                self._fabric_instance.launch()
            # Set the log directory in config if it is not already set
            if (
                self.log_dir is not None
                and hasattr(config, "log_dir")
                and config.get("log_dir", None) is None
            ):
                if self._fabric_instance.is_global_zero:
                    log.info(f"Setting log_dir to {self.log_dir}")
                config.log_dir = self.log_dir

    @property
    def fabric(self):
        if self._fabric_instance is None:
            self.setup_lightning_fabric(getattr(self, "config", DictConfig({})))
        return self._fabric_instance

    @property
    def log_dir(self):
        """
        Retrieves the log directory from the fabric's logger.
        """
        if self.fabric is not None and len(self.fabric._loggers) > 0:
            log_dir = self.fabric.logger.log_dir
            if self.fabric.is_global_zero and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            return log_dir
        else:
            return None

    def to_device(self, obj: TensorOrModule) -> TensorOrModule:
        """
        Moves a tensor or module to the proper device.

        Args:
            obj (TensorOrModule): The tensor or module to move to the device.

        Returns:
            TensorOrModule: the same type of object as the input, moved to the device.
        """
        return self.fabric.to_device(obj)

    @rank_zero_only
    def log_hyperparams(
        self,
        config: Optional[DictConfig] = None,
        save_dir: Optional[str] = None,
        filename: str = "config.yaml",
    ):
        R"""
        Logs the hyperparameters and saves the configuration to a YAML file.
        The YAML file is saved in the log directory by default with the name `config.yaml`, or in the specified save directory `save_dir`.

        Args:
            config (Optional[DictConfig]): The configuration to log and save. If not provided, the class's `config` attribute is used.
            save_dir (Optional[str]): The directory in which to save the configuration file. If not provided, the log directory is used.
            filename (str): The name of the configuration file. Default is `config.yaml`.
        """
        if config is None:
            config = self.config
        if save_dir is None:
            save_dir = self.log_dir
        self.fabric.logger.log_hyperparams(
            OmegaConf.to_container(config, resolve=True, enum_to_str=True)
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(self.log_dir if save_dir is None else save_dir, filename),
        )

    @property
    def tensorboard_summarywriter(
        self,
    ) -> "lightning.fabric.loggers.tensorboard.SummaryWriter":
        if isinstance(self.fabric.logger, TensorBoardLogger):
            return self.fabric.logger.experiment
        else:
            raise AttributeError("the logger is not a TensorBoardLogger.")

    @property
    def is_debug_mode(self):
        if hasattr(self, "config") and self.config.get("fast_dev_run", False):
            return True
        elif hasattr(self, "_program") and self._program.config.get(
            "fast_dev_run", False
        ):
            return True
        else:
            return False
