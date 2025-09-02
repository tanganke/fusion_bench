import inspect
import logging
from copy import deepcopy
from functools import wraps
from inspect import Parameter, _ParameterKind
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from bidict import MutableBidict, bidict
from omegaconf import DictConfig, OmegaConf

from fusion_bench.constants import FUSION_BENCH_VERSION
from fusion_bench.utils import import_object, instantiate
from fusion_bench.utils.instantiate_utils import set_print_function_call

log = logging.getLogger(__name__)

__all__ = [
    "auto_register_config",
    "YAMLSerializationMixin",
    "BaseYAMLSerializable",
]


def _set_attr(self, param_name: str, value):
    """
    Set an attribute on the object using the parameter name from config mapping.

    This function looks up the corresponding attribute name for the given parameter
    name using the object's _config_mapping, then sets that attribute to the
    specified value. It also logs the operation for debugging purposes.

    Args:
        self: The object instance to set the attribute on.
        param_name (str): The parameter name (config key) to map to an attribute.
        value: The value to assign to the attribute.

    Raises:
        ValueError: If the parameter name is not found in the config mapping.
    """
    attr_name = self._config_mapping.inverse[param_name]
    log.debug(f"set {attr_name} to {value}. Parameter name: {param_name}")
    setattr(self, attr_name, value)


def auto_register_config(cls):
    """
    Decorator to automatically register __init__ parameters in _config_mapping.

    This decorator enhances classes that inherit from YAMLSerializationMixin by
    automatically mapping constructor parameters to configuration keys and
    dynamically setting instance attributes based on provided arguments.

    The decorator performs the following operations:
    1. Inspects the class's __init__ method signature
    2. Automatically populates the _config_mapping dictionary with parameter names
    3. Wraps the __init__ method to handle both positional and keyword arguments
    4. Sets instance attributes for all constructor parameters
    5. Applies default values when parameters are not provided

    Args:
        cls (YAMLSerializationMixin): The class to be decorated. Must inherit from
            YAMLSerializationMixin to ensure proper serialization capabilities.

    Returns:
        YAMLSerializationMixin: The decorated class with enhanced auto-registration
            functionality and modified __init__ behavior.

    Behavior:
        - **Parameter Registration**: All non-variadic parameters (excluding ``*args``, ``**kwargs``)
          from the __init__ method are automatically added to _config_mapping
        - **Positional Arguments**: Handled in order and mapped to corresponding parameter names
        - **Keyword Arguments**: Processed after positional arguments, overriding any conflicts
        - **Default Values**: Applied when parameters are not provided via arguments
        - **Attribute Setting**: All parameters become instance attributes accessible via dot notation

    Note:
        - The decorator wraps the original __init__ method while preserving its signature for IDE support
        - Parameters with ``*args`` or ``**kwargs`` signatures are ignored during registration
        - The attributes are auto-registered, then the original __init__ method is called,
        - Type hints, method name, and other metadata are preserved using functools.wraps
        - This decorator is designed to work seamlessly with the YAML serialization system

    Raises:
        AttributeError: If the class does not have the required _config_mapping attribute
            infrastructure (should inherit from YAMLSerializationMixin)
    """
    original_init = cls.__init__
    sig = inspect.signature(original_init)

    # Auto-register parameters in _config_mapping
    if not "_config_mapping" in cls.__dict__:
        cls._config_mapping = deepcopy(getattr(cls, "_config_mapping", bidict()))
    if not isinstance(cls._config_mapping, bidict):
        cls._config_mapping = bidict(cls._config_mapping)

    registered_parameters = tuple(cls._config_mapping.values())

    for param_name in list(sig.parameters.keys())[1:]:  # Skip 'self'
        if (
            sig.parameters[param_name].kind
            not in [
                _ParameterKind.VAR_POSITIONAL,
                _ParameterKind.VAR_KEYWORD,
            ]
        ) and (param_name not in registered_parameters):
            cls._config_mapping[param_name] = param_name

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        log.debug(f"set attributes for {self.__class__.__name__} in {cls.__name__}")
        # auto-register the attributes based on the signature
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'

        # Handle positional arguments
        for i, arg_value in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                if sig.parameters[param_name].kind not in [
                    _ParameterKind.VAR_POSITIONAL,
                    _ParameterKind.VAR_KEYWORD,
                ]:
                    _set_attr(self, param_name, arg_value)

        # Handle keyword arguments and defaults
        for param_name in param_names:
            if sig.parameters[param_name].kind not in [
                _ParameterKind.VAR_POSITIONAL,
                _ParameterKind.VAR_KEYWORD,
            ]:
                # Skip if already set by positional argument
                param_index = param_names.index(param_name)
                if param_index >= 0 and param_index < len(args):
                    continue

                if param_name in kwargs:
                    _set_attr(self, param_name, kwargs[param_name])
                else:
                    # Set default value if available and attribute doesn't exist
                    default_value = sig.parameters[param_name].default
                    if default_value is not Parameter.empty:
                        _set_attr(self, param_name, default_value)

        # Call the original __init__
        result = original_init(self, *args, **kwargs)
        return result

    # Replace the original __init__ method while preserving its signature
    cls.__init__ = __init__
    return cls


class YAMLSerializationMixin:
    _config_key: Optional[str] = None
    _config_mapping: MutableBidict[str, str] = bidict()
    R"""
    `_config_mapping` is a dictionary mapping the attribute names of the class to the config option names. This is used to convert the class to a DictConfig.

    >>> algorithm.config
        DictCOnfig({'_target_': 'SomeModelFusionAlgorithm', 'hyper_param_1': 1, 'hyper_param_2': 2})

    By default, the `_target_` key is set to the class name as `type(self).__name__`.
    """

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            log.warning(f"Unused argument: {key}={value}")

    @property
    def config(self) -> DictConfig:
        R"""
        Returns the configuration of the model pool as a DictConfig.

        This property converts the model pool instance into a dictionary
        configuration, which can be used for serialization or other purposes.

        Returns:
            DictConfig: The configuration of the model pool.
        """
        config = {"_target_": f"{type(self).__module__}.{type(self).__qualname__}"}
        for attr, key in self._config_mapping.items():
            if hasattr(self, attr):
                config[key] = getattr(self, attr)

        try:
            return OmegaConf.create(config)
        except Exception as e:
            return OmegaConf.create(config, flags={"allow_objects": True})

    def to_yaml(self, path: Union[str, Path], resolve: bool = True):
        """
        Save the model pool to a YAML file.

        Args:
            path (Union[str, Path]): The path to save the model pool to.
        """
        OmegaConf.save(self.config, path, resolve=resolve)

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
        with set_print_function_call(False):
            return instantiate(config)

    def register_parameter_to_config(
        self,
        attr_name: str,
        param_name: str,
        value,
    ):
        """
        Set an attribute value and register its config mapping.

        This method allows dynamic setting of object attributes while simultaneously
        updating the configuration mapping that defines how the attribute should
        be serialized in the configuration output.

        Args:
            attr_name (str): The name of the attribute to set on this object.
            arg_name (str): The corresponding parameter name to use in the config
                serialization. This is how the attribute will appear in YAML output.
            value: The value to assign to the attribute.

        """
        setattr(self, attr_name, value)
        self._config_mapping[attr_name] = param_name


@auto_register_config
class BaseYAMLSerializable(YAMLSerializationMixin):
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
        class MyAlgorithm(BaseYAMLSerializable):
            _config_mapping = BaseYAMLSerializable._config_mapping | {
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

    def __init__(
        self,
        _recursive_: bool = False,
        _usage_: Optional[str] = None,
        _version_: Optional[str] = FUSION_BENCH_VERSION,
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
            model = BaseYAMLSerializable(
                _usage_="Image classification on CIFAR-10",
                _version_="2.1.0"
            )
            ```
        """
        super().__init__(**kwargs)
        if _version_ != FUSION_BENCH_VERSION:
            log.warning(
                f"Current fusion-bench version is {FUSION_BENCH_VERSION}, but the serialized version is {_version_}. "
                "Attempting to use current version."
            )
            # override _version_ with current fusion-bench version
            self._version_ = FUSION_BENCH_VERSION
