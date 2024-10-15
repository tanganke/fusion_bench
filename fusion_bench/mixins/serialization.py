import logging
from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import OmegaConf

from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)


class YAMLSerializationMixin:
    _recursive_: bool = False
    _config_mapping: Dict[str, str] = {
        "_recursive_": "_recursive_",
    }
    R"""
    `_config_mapping` is a dictionary mapping the attribute names of the class to the config option names. This is used to convert the class to a DictConfig.

    For example, if an algorithm class is defined as follows:
    
    ```python
    class SomeModelFusionAlgorithm(BaseModelFusionAlgorithm):
        hyper_parameter_1 = None
        hyper_parameter_2 = None

        _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
            "hyper_parameter_1" : "hyper_param_1",
            "hyper_parameter_2" : "hyper_param_2",
        }
        def __init__(self, hyper_param_1: int, hyper_param_2: int):
            self.hyper_parameter_1 = hyper_param_1
            self.hyper_parameter_2 = hyper_param_2
            super().__init__()
    ```

    The model pool will be converted to a DictConfig as follows:
        
    ```python
    algorithm = SomeModelFusionAlgorithm(hyper_param_1=1, hyper_param_2=2)
    ```

    >>> algorithm.config
        DictCOnfig({'_target_': 'SomeModelFusionAlgorithm', 'hyper_param_1': 1, 'hyper_param_2': 2})

    By default, the `_target_` key is set to the class name as `type(self).__name__`.
    """

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
        R"""
        Returns the configuration of the model pool as a DictConfig.

        This property calls the `to_config` method to convert the model pool
        instance into a dictionary configuration, which can be used for
        serialization or other purposes.

        Example:
            >>> model = SomeModelFusionAlgorithm(hyper_param_1=1, hyper_param_2=2)
            >>> config = model.config
            >>> print(config)
            DictConfig({'_target_': 'SomeModelFusionAlgorithm', 'hyper_param_1': 1, 'hyper_param_2': 2})

        This is useful for serializing the object to a YAML file or for debugging.

        Returns:
            DictConfig: The configuration of the model pool.
        """
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
