from .timer import timeit_context
from .parameters import print_parameters
import importlib


def import_class(abs_class_name):
    """
    Imports a class from a module given the absolute class name.

    Args:
        abs_class_name (str): The absolute name of the class to import.

    Returns:
        The imported class.
    """
    module_name, class_name = abs_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
