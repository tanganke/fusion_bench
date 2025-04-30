import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version

if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_pyav_available():
    return _is_package_available("av")


def is_fastapi_available():
    return _is_package_available("fastapi")


def is_galore_available():
    return _is_package_available("galore_torch")


def is_gradio_available():
    return _is_package_available("gradio")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_pillow_available():
    return _is_package_available("PIL")


def is_requests_available():
    return _is_package_available("requests")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


def is_starlette_available():
    return _is_package_available("sse_starlette")


@lru_cache
def is_transformers_version_greater_than_4_43():
    return _get_package_version("transformers") >= version.parse("4.43.0")


def is_uvicorn_available():
    return _is_package_available("uvicorn")


def is_vllm_available():
    return _is_package_available("vllm")


def import_object(abs_obj_name: str):
    """
    Imports a class from a module given the absolute class name.

    Args:
        abs_obj_name (str): The absolute name of the object to import.

    Returns:
        The imported class.
    """
    module_name, obj_name = abs_obj_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def compare_versions(v1, v2):
    """Compare two version strings.
    Returns -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2"""

    v1 = version.parse(v1)
    v2 = version.parse(v2)
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0
