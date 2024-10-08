import functools
from typing import Any, Callable, Tuple, Union

from fusion_bench.utils import import_object


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
