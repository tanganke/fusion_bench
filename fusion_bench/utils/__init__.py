import importlib

from . import data
from .cache_utils import *
from .devices import *
from .dtype import parse_dtype
from .instantiate import instantiate
from .parameters import *
from .timer import timeit_context


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
