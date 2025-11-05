# flake8: noqa: F401
import sys
from typing import TYPE_CHECKING

from . import functools
from .lazy_imports import LazyImporter

_extra_objects = {
    "functools": functools,
}
_import_structure = {
    "cache_utils": [
        "cache_to_disk",
        "cache_with_joblib",
        "set_default_cache_dir",
    ],
    "data": [
        "InfiniteDataLoader",
        "load_tensor_from_file",
        "train_validation_split",
        "train_validation_test_split",
    ],
    "devices": [
        "clear_cuda_cache",
        "get_current_device",
        "get_device",
        "get_device_capabilities",
        "get_device_memory_info",
        "num_devices",
        "to_device",
    ],
    "dtype": ["get_dtype", "parse_dtype"],
    "fabric": ["seed_everything_by_time"],
    "instantiate_utils": [
        "instantiate",
        "is_instantiable",
        "set_print_function_call",
        "set_print_function_call_permeanent",
    ],
    "json": ["load_from_json", "save_to_json", "print_json"],
    "lazy_state_dict": ["LazyStateDict"],
    "misc": [
        "first",
        "has_length",
        "join_lists",
        "validate_and_suggest_corrections",
    ],
    "packages": ["compare_versions", "import_object"],
    "parameters": [
        "check_parameters_all_equal",
        "count_parameters",
        "get_parameter_statistics",
        "get_parameter_summary",
        "human_readable",
        "print_parameters",
        "state_dict_to_vector",
        "trainable_state_dict",
        "vector_to_state_dict",
    ],
    "path": [
        "create_symlink",
        "listdir_fullpath",
        "path_is_dir_and_not_empty",
    ],
    "pylogger": [
        "RankedLogger",
        "RankZeroLogger",
        "get_rankzero_logger",
    ],
    "state_dict_arithmetic": [
        "ArithmeticStateDict",
        "state_dicts_check_keys",
        "num_params_of_state_dict",
        "state_dict_to_device",
        "state_dict_flatten",
        "state_dict_avg",
        "state_dict_sub",
        "state_dict_add",
        "state_dict_add_scalar",
        "state_dict_mul",
        "state_dict_div",
        "state_dict_power",
        "state_dict_interpolation",
        "state_dict_sum",
        "state_dict_weighted_sum",
        "state_dict_diff_abs",
        "state_dict_binary_mask",
        "state_dict_hadamard_product",
    ],
    "timer": ["timeit_context"],
    "type": [
        "BoolStateDictType",
        "StateDictType",
        "TorchModelType",
    ],
    "validation": [
        "validate_path_exists",
        "validate_file_exists",
        "validate_directory_exists",
        "validate_model_name",
        "ValidationError",
    ],
}

if TYPE_CHECKING:
    from .cache_utils import cache_to_disk, cache_with_joblib, set_default_cache_dir
    from .data import (
        InfiniteDataLoader,
        load_tensor_from_file,
        train_validation_split,
        train_validation_test_split,
    )
    from .devices import (
        clear_cuda_cache,
        get_current_device,
        get_device,
        get_device_capabilities,
        get_device_memory_info,
        num_devices,
        to_device,
    )
    from .dtype import get_dtype, parse_dtype
    from .fabric import seed_everything_by_time
    from .instantiate_utils import (
        instantiate,
        is_instantiable,
        set_print_function_call,
        set_print_function_call_permeanent,
    )
    from .json import load_from_json, print_json, save_to_json
    from .lazy_state_dict import LazyStateDict
    from .misc import first, has_length, join_lists, validate_and_suggest_corrections
    from .packages import compare_versions, import_object
    from .parameters import (
        check_parameters_all_equal,
        count_parameters,
        get_parameter_statistics,
        get_parameter_summary,
        human_readable,
        print_parameters,
        state_dict_to_vector,
        trainable_state_dict,
        vector_to_state_dict,
    )
    from .path import create_symlink, listdir_fullpath, path_is_dir_and_not_empty
    from .pylogger import RankedLogger, RankZeroLogger, get_rankzero_logger
    from .state_dict_arithmetic import (
        ArithmeticStateDict,
        num_params_of_state_dict,
        state_dict_add,
        state_dict_add_scalar,
        state_dict_avg,
        state_dict_binary_mask,
        state_dict_diff_abs,
        state_dict_div,
        state_dict_flatten,
        state_dict_hadamard_product,
        state_dict_interpolation,
        state_dict_mul,
        state_dict_power,
        state_dict_sub,
        state_dict_sum,
        state_dict_to_device,
        state_dict_weighted_sum,
        state_dicts_check_keys,
    )
    from .timer import timeit_context
    from .type import BoolStateDictType, StateDictType, TorchModelType
    from .validation import (
        ValidationError,
        validate_directory_exists,
        validate_file_exists,
        validate_model_name,
        validate_path_exists,
    )

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects=_extra_objects,
    )
