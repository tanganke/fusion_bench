import importlib

from .parameters import print_parameters
from .timer import timeit_context


def import_object(abs_obj_name):
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
