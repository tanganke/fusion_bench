# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from Hydra
import copy
import functools
from contextlib import contextmanager
from enum import Enum
from textwrap import dedent
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from hydra._internal.utils import _locate
from hydra.errors import InstantiationException
from hydra.types import ConvertMode, TargetConf
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, SCMode
from omegaconf._utils import is_structured_config
from rich import print
from rich.panel import Panel
from rich.syntax import Syntax

PRINT_FUNCTION_CALL = True
"""
Flag to determine whether function calls should be printed.
If set to True, function calls will be printed.
"""

PRINT_FUNCTION_CALL_FUNC = print
"""
Function to be used for printing function calls.
"""

CATCH_EXCEPTION = True


@contextmanager
def set_print_function_call(value: bool):
    global PRINT_FUNCTION_CALL
    old_value = PRINT_FUNCTION_CALL
    PRINT_FUNCTION_CALL = value
    try:
        yield
    finally:
        PRINT_FUNCTION_CALL = old_value

def set_print_function_call_permeanent(value: bool):
    global PRINT_FUNCTION_CALL
    PRINT_FUNCTION_CALL = value

def is_instantiable(config: Union[DictConfig, Any]) -> bool:
    if OmegaConf.is_dict(config):
        return "_target_" in config
    return False


def _resolve_callable_name(f: Callable[..., Any]) -> str:
    # Get the module name
    module_name = f.__module__
    # Get the qualname (fully qualified name) of the callable
    qualname = f.__qualname__
    # If the module is not "__main__", include it in the full name
    if module_name != "__main__":
        full_name = f"{module_name}.{qualname}"
    else:
        full_name = qualname

    return full_name


def _format_args_kwargs(args, kwargs):
    result_strings = []
    if len(args) > 0:
        result_strings.append(", ".join(repr(arg) for arg in args))
    if len(kwargs) > 0:
        result_strings.append(", ".join(f"{k}={repr(v)}" for k, v in kwargs.items()))

    if len(result_strings) == 0:
        return ""
    else:
        return ", ".join(result_strings)


class _Keys(str, Enum):
    """Special keys in configs used by instantiate."""

    TARGET = "_target_"
    CONVERT = "_convert_"
    RECURSIVE = "_recursive_"
    ARGS = "_args_"
    PARTIAL = "_partial_"


def _is_target(x: Any) -> bool:
    if isinstance(x, dict):
        return "_target_" in x
    if OmegaConf.is_dict(x):
        return "_target_" in x
    return False


def _extract_pos_args(input_args: Any, kwargs: Any) -> Tuple[Any, Any]:
    config_args = kwargs.pop(_Keys.ARGS, ())
    output_args = config_args

    if isinstance(config_args, Sequence):
        if len(input_args) > 0:
            output_args = input_args
    else:
        raise InstantiationException(
            f"Unsupported _args_ type: '{type(config_args).__name__}'. value: '{config_args}'"
        )

    return output_args, kwargs


def _call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    try:
        args, kwargs = _extract_pos_args(args, kwargs)
        # detaching configs from parent.
        # At this time, everything is resolved and the parent link can cause
        # issues when serializing objects in some scenarios.
        for arg in args:
            if OmegaConf.is_config(arg):
                arg._set_parent(None)
        for v in kwargs.values():
            if OmegaConf.is_config(v):
                v._set_parent(None)
    except Exception as e:
        msg = (
            f"Error in collecting args and kwargs for '{_convert_target_to_string(_target_)}':"
            + f"\n{repr(e)}"
        )
        if full_key:
            msg += f"\nfull_key: {full_key}"

        raise InstantiationException(msg) from e

    if _partial_:
        if PRINT_FUNCTION_CALL and getattr(rank_zero_only, "rank", 0) == 0:
            call_str = f"functools.partial({_resolve_callable_name(_target_)}, {_format_args_kwargs(args, kwargs)})"
            PRINT_FUNCTION_CALL_FUNC(
                Panel(
                    Syntax(call_str, "python", theme="monokai", word_wrap=True),
                    title="Instantiate by calling partial",
                    border_style="cyan",
                )
            )

        if CATCH_EXCEPTION:
            try:
                return functools.partial(_target_, *args, **kwargs)
            except Exception as e:
                msg = (
                    f"Error in creating partial({_convert_target_to_string(_target_)}, ...) object:"
                    + f"\n{repr(e)}"
                )
                if full_key:
                    msg += f"\nfull_key: {full_key}"
                raise InstantiationException(msg) from e
        else:
            return functools.partial(_target_, *args, **kwargs)
    else:
        if PRINT_FUNCTION_CALL and getattr(rank_zero_only, "rank", 0) == 0:
            call_str = f"{_resolve_callable_name(_target_)}({_format_args_kwargs(args, kwargs)})"
            PRINT_FUNCTION_CALL_FUNC(
                Panel(
                    Syntax(call_str, "python", theme="monokai", word_wrap=True),
                    title="Instantiate by calling function",
                    border_style="green",
                )
            )
        if CATCH_EXCEPTION:
            try:
                return _target_(*args, **kwargs)
            except Exception as e:
                msg = f"Error in call to target '{_convert_target_to_string(_target_)}':\n{repr(e)}"
                if full_key:
                    msg += f"\nfull_key: {full_key}"
                raise InstantiationException(msg) from e
        else:
            return _target_(*args, **kwargs)


def _convert_target_to_string(t: Any) -> Any:
    if callable(t):
        return f"{t.__module__}.{t.__qualname__}"
    else:
        return t


def _prepare_input_dict_or_list(d: Union[Dict[Any, Any], List[Any]]) -> Any:
    res: Any
    if isinstance(d, dict):
        res = {}
        for k, v in d.items():
            if k == "_target_":
                v = _convert_target_to_string(d["_target_"])
            elif isinstance(v, (dict, list)):
                v = _prepare_input_dict_or_list(v)
            res[k] = v
    elif isinstance(d, list):
        res = []
        for v in d:
            if isinstance(v, (list, dict)):
                v = _prepare_input_dict_or_list(v)
            res.append(v)
    else:
        assert False
    return res


def _resolve_target(
    target: Union[str, type, Callable[..., Any]], full_key: str
) -> Union[type, Callable[..., Any]]:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        try:
            target = _locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}', set env var HYDRA_FULL_ERROR=1 to see chained exception."
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    if not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)
    return target


def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    :param config: An config object describing what to call and what params to use.
                   In addition to the parameters, the config must contain:
                   _target_ : target class or callable name (str)
                   And may contain:
                   _args_: List-like of positional arguments to pass to the target
                   _recursive_: Construct nested objects as well (bool).
                                True by default.
                                may be overridden via a _recursive_ key in
                                the kwargs
                   _convert_: Conversion strategy
                        none    : Passed objects are DictConfig and ListConfig, default
                        partial : Passed objects are converted to dict and list, with
                                  the exception of Structured Configs (and their fields).
                        object  : Passed objects are converted to dict and list.
                                  Structured Configs are converted to instances of the
                                  backing dataclass / attr class.
                        all     : Passed objects are dicts, lists and primitives without
                                  a trace of OmegaConf containers. Structured configs
                                  are converted to dicts / lists too.
                   _partial_: If True, return functools.partial wrapped method or object
                              False by default. Configure per target.
    :param args: Optional positional parameters pass-through
    :param kwargs: Optional named parameters to override
                   parameters in the config object. Parameters not present
                   in the config objects are being passed as is to the target.
                   IMPORTANT: dataclasses instances in kwargs are interpreted as config
                              and cannot be used as passthrough
    :return: if _target_ is a class name: the instantiated object
             if _target_ is a callable: the return value of the call
    """

    # Return None if config is None
    if config is None:
        return None

    # TargetConf edge case
    if isinstance(config, TargetConf) and config._target_ == "???":
        # Specific check to give a good warning about failure to annotate _target_ as a string.
        raise InstantiationException(
            dedent(
                f"""\
                Config has missing value for key `_target_`, cannot instantiate.
                Config type: {type(config).__name__}
                Check that the `_target_` key in your dataclass is properly annotated and overridden.
                A common problem is forgetting to annotate _target_ as a string : '_target_: str = ...'"""
            )
        )
        # TODO: print full key

    if isinstance(config, (dict, list)):
        config = _prepare_input_dict_or_list(config)

    kwargs = _prepare_input_dict_or_list(kwargs)

    # Structured Config always converted first to OmegaConf
    if is_structured_config(config) or isinstance(config, (dict, list)):
        config = OmegaConf.structured(config, flags={"allow_objects": True})

    if OmegaConf.is_dict(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(
            flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
        )
        config_copy._set_parent(config._get_parent())
        config = config_copy

        if kwargs:
            config = OmegaConf.merge(config, kwargs)

        OmegaConf.resolve(config)

        _recursive_ = config.pop(_Keys.RECURSIVE, True)
        _convert_ = config.pop(_Keys.CONVERT, ConvertMode.NONE)
        _partial_ = config.pop(_Keys.PARTIAL, False)

        return instantiate_node(
            config, *args, recursive=_recursive_, convert=_convert_, partial=_partial_
        )
    elif OmegaConf.is_list(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(
            flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
        )
        config_copy._set_parent(config._get_parent())
        config = config_copy

        OmegaConf.resolve(config)

        _recursive_ = kwargs.pop(_Keys.RECURSIVE, True)
        _convert_ = kwargs.pop(_Keys.CONVERT, ConvertMode.NONE)
        _partial_ = kwargs.pop(_Keys.PARTIAL, False)

        if _partial_:
            raise InstantiationException(
                "The _partial_ keyword is not compatible with top-level list instantiation"
            )

        return instantiate_node(
            config, *args, recursive=_recursive_, convert=_convert_, partial=_partial_
        )
    else:
        raise InstantiationException(
            dedent(
                f"""\
                Cannot instantiate config of type {type(config).__name__}.
                Top level config must be an OmegaConf DictConfig/ListConfig object,
                a plain dict/list, or a Structured Config class or instance."""
            )
        )


def _convert_node(node: Any, convert: Union[ConvertMode, str]) -> Any:
    if OmegaConf.is_config(node):
        if convert == ConvertMode.ALL:
            node = OmegaConf.to_container(node, resolve=True)
        elif convert == ConvertMode.PARTIAL:
            node = OmegaConf.to_container(
                node, resolve=True, structured_config_mode=SCMode.DICT_CONFIG
            )
        elif convert == ConvertMode.OBJECT:
            node = OmegaConf.to_container(
                node, resolve=True, structured_config_mode=SCMode.INSTANTIATE
            )
    return node


def instantiate_node(
    node: Any,
    *args: Any,
    convert: Union[str, ConvertMode] = ConvertMode.NONE,
    recursive: bool = True,
    partial: bool = False,
) -> Any:
    # Return None if config is None
    if node is None or (OmegaConf.is_config(node) and node._is_none()):
        return None

    if not OmegaConf.is_config(node):
        return node

    # Override parent modes from config if specified
    if OmegaConf.is_dict(node):
        # using getitem instead of get(key, default) because OmegaConf will raise an exception
        # if the key type is incompatible on get.
        convert = node[_Keys.CONVERT] if _Keys.CONVERT in node else convert
        recursive = node[_Keys.RECURSIVE] if _Keys.RECURSIVE in node else recursive
        partial = node[_Keys.PARTIAL] if _Keys.PARTIAL in node else partial

    full_key = node._get_full_key(None)

    if not isinstance(recursive, bool):
        msg = f"Instantiation: _recursive_ flag must be a bool, got {type(recursive)}"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise TypeError(msg)

    if not isinstance(partial, bool):
        msg = f"Instantiation: _partial_ flag must be a bool, got {type(partial)}"
        if node and full_key:
            msg += f"\nfull_key: {full_key}"
        raise TypeError(msg)

    # If OmegaConf list, create new list of instances if recursive
    if OmegaConf.is_list(node):
        items = [
            instantiate_node(item, convert=convert, recursive=recursive)
            for item in node._iter_ex(resolve=True)
        ]

        if convert in (ConvertMode.ALL, ConvertMode.PARTIAL, ConvertMode.OBJECT):
            # If ALL or PARTIAL or OBJECT, use plain list as container
            return items
        else:
            # Otherwise, use ListConfig as container
            lst = OmegaConf.create(items, flags={"allow_objects": True})
            lst._set_parent(node)
            return lst

    elif OmegaConf.is_dict(node):
        exclude_keys = set({"_target_", "_convert_", "_recursive_", "_partial_"})
        if _is_target(node):
            _target_ = _resolve_target(node.get(_Keys.TARGET), full_key)
            kwargs = {}
            is_partial = node.get("_partial_", False) or partial
            for key in node.keys():
                if key not in exclude_keys:
                    if OmegaConf.is_missing(node, key) and is_partial:
                        continue
                    value = node[key]
                    if recursive:
                        value = instantiate_node(
                            value, convert=convert, recursive=recursive
                        )
                    kwargs[key] = _convert_node(value, convert)

            return _call_target(_target_, partial, args, kwargs, full_key)
        else:
            # If ALL or PARTIAL non structured or OBJECT non structured,
            # instantiate in dict and resolve interpolations eagerly.
            if convert == ConvertMode.ALL or (
                convert in (ConvertMode.PARTIAL, ConvertMode.OBJECT)
                and node._metadata.object_type in (None, dict)
            ):
                dict_items = {}
                for key, value in node.items():
                    # list items inherits recursive flag from the containing dict.
                    dict_items[key] = instantiate_node(
                        value, convert=convert, recursive=recursive
                    )
                return dict_items
            else:
                # Otherwise use DictConfig and resolve interpolations lazily.
                cfg = OmegaConf.create({}, flags={"allow_objects": True})
                for key, value in node.items():
                    cfg[key] = instantiate_node(
                        value, convert=convert, recursive=recursive
                    )
                cfg._set_parent(node)
                cfg._metadata.object_type = node._metadata.object_type
                if convert == ConvertMode.OBJECT:
                    return OmegaConf.to_object(cfg)
                return cfg

    else:
        assert False, f"Unexpected config type : {type(node).__name__}"
