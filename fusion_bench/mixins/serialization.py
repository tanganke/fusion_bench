import logging
from pathlib import Path
from typing import Dict, Optional, Union

from hydra.utils import instantiate
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


class YAMLSerializationMixin:
    _recursive_: bool = False
    _config_mapping: Dict[str, str] = {
        "_recursive_": "_recursive_",
    }

    def __init__(
        self,
        _recursive_: bool = False,
        **kwargs,
    ) -> None:
        self._recursive_ = _recursive_
        for key, value in kwargs.items():
            log.warning(f"Unused argument: {key}={value}")

    @property
    def config(self):
        return self.to_config()

    def to_yaml(self, path: Union[str, Path]):
        """
        Save the model pool to a YAML file.

        Args:
            path (Union[str, Path]): The path to save the model pool to.
        """
        config = self.to_config()
        OmegaConf.save(config, path, resolve=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]):
        """
        Load a model pool from a YAML file.

        Args:
            path (Union[str, Path]): The path to load the model pool from.

        Returns:
            BaseModelPool: The loaded model pool.
        """
        config = OmegaConf.load(path)
        return instantiate(config, _recursive_=cls._recursive_)

    def to_config(self):
        """
        Convert the model pool to a DictConfig.

        Returns:
            Dict: The model pool as a DictConfig.
        """
        config = {"_target_": type(self).__name__}
        for attr, key in self._config_mapping.items():
            if hasattr(self, attr):
                config[key] = getattr(self, attr)
        return OmegaConf.create(config)


class BaseYAMLSerializableModel(YAMLSerializationMixin):
    _config_mapping = YAMLSerializationMixin._config_mapping | {
        "_usage_": "_usage_",
        "_version_": "_version_",
    }

    def __init__(
        self,
        _usage_: Optional[str] = None,
        _version_: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._usage_ = _usage_
        self._version_ = _version_
