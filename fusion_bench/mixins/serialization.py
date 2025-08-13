import logging
from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from fusion_bench.utils import import_object, instantiate

log = logging.getLogger(__name__)


class YAMLSerializationMixin:
    _recursive_: bool = False
    _config_key: Optional[str] = None
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
    def config(self) -> DictConfig:
        R"""
        Returns the configuration of the model pool as a DictConfig.

        This property calls the `to_config` method to convert the model pool
        instance into a dictionary configuration, which can be used for
        serialization or other purposes.

        Example:

        ```python
        model = SomeModelFusionAlgorithm(hyper_param_1=1, hyper_param_2=2)
        config = model.config
        print(config)
        # DictConfig({'_target_': 'SomeModelFusionAlgorithm', 'hyper_param_1': 1, 'hyper_param_2': 2})
        ```

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
        if cls._config_key is not None and cls._config_key in config:
            config = config[cls._config_key]
        target_cls = import_object(config["_target_"])
        if target_cls != cls:
            log.warning(
                f"The class {target_cls.__name__} is not the same as the class {cls.__name__}. "
                f"Instantiating the class {target_cls.__name__} instead."
            )
        return instantiate(
            config,
            _recursive_=(
                cls._recursive_
                if config.get("_recursive_") is None
                else config.get("_recursive_")
            ),
        )

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
    """
    A base class for YAML-serializable classes with enhanced metadata support.

    This class extends `YAMLSerializationMixin` to provide additional metadata
    fields commonly used in FusionBench classes, including usage information
    and version tracking. It serves as a foundation for all serializable
    model components in the framework.

    The class automatically handles serialization of usage and version metadata
    alongside the standard configuration parameters, making it easier to track
    model provenance and intended usage patterns.

    Attributes:
        _usage_ (Optional[str]): Description of the model's intended usage or purpose.
        _version_ (Optional[str]): Version information for the model or configuration.

    Example:
        ```python
        class MyAlgorithm(BaseYAMLSerializableModel):
            _config_mapping = BaseYAMLSerializableModel._config_mapping | {
                "model_name": "model_name",
                "num_layers": "num_layers",
            }

            def __init__(self, _usage_: str = None, _version_: str = None):
                super().__init__(_usage_=_usage_, _version_=_version_)

        # Usage with metadata
        model = MyAlgorithm(
            _usage_="Text classification fine-tuning",
            _version_="1.0.0"
        )

        # Serialization includes metadata
        config = model.config
        # DictConfig({
        #     '_target_': 'MyModel',
        #     '_usage_': 'Text classification fine-tuning',
        #     '_version_': '1.0.0'
        # })
        ```

    Note:
        The underscore prefix in `_usage_` and `_version_` follows the convention
        for metadata fields that are not core model parameters but provide
        important contextual information for model management and tracking.
    """

    _config_mapping = YAMLSerializationMixin._config_mapping | {
        "_usage_": "_usage_",
        "_version_": "_version_",
    }

    _usage_: Optional[str] = None
    _version_: Optional[str] = None

    def __init__(
        self,
        _usage_: Optional[str] = None,
        _version_: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a base YAML-serializable model with metadata support.

        Args:
            _usage_ (Optional[str], optional): Description of the model's intended
                usage or purpose. This can include information about the training
                domain, expected input types, or specific use cases. Defaults to None.
            _version_ (Optional[str], optional): Version information for the model
                or configuration. Can be used to track model iterations, dataset
                versions, or compatibility information. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
                Unused arguments will trigger warnings via the parent's initialization.

        Example:
            ```python
            model = BaseYAMLSerializableModel(
                _usage_="Image classification on CIFAR-10",
                _version_="2.1.0"
            )
            ```
        """
        super().__init__(**kwargs)
        if _usage_ is not None:
            self._usage_ = _usage_
        if _version_ is not None:
            self._version_ = _version_
