import functools
import inspect
from typing import Callable, Union

from .packages import import_object


@functools.cache
def cached_func_call(func: Union[str, Callable], *args, **kwargs):
    R"""
    Calls the given function with the provided arguments and caches the result.

    This function uses `functools.cache` to cache the result of the function call,
    so that subsequent calls with the same arguments return the cached result
    instead of recomputing it.

    Args:
        func (Union[str, Callable]): The function to be called. If `func` is a string, it is assumed to be the absolute name of the function to be imported.
        *args: Variable length argument list to be passed to the function.
        **kwargs: Arbitrary keyword arguments to be passed to the function.

    Returns:
        The result of the function call.
    """
    if isinstance(func, str):
        func = import_object(func)
    return func(*args, **kwargs)


def number_of_arguments(func):
    """
    Return the number of arguments of the passed function, even if it's a partial function.
    """
    if isinstance(func, functools.partial):
        total_args = len(inspect.signature(func.func).parameters)
        return total_args - len(func.args) - len(func.keywords)
    return len(inspect.signature(func).parameters)
