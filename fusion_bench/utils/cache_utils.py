import logging
import os
import pickle
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

from joblib import Memory

__all__ = ["cache_to_disk", "cache_with_joblib", "set_default_cache_dir"]


log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.cwd() / "outputs" / "cache"


def set_default_cache_dir(path: str | Path):
    global DEFAULT_CACHE_DIR
    if path is None:
        return

    if isinstance(path, str):
        path = Path(path)
    DEFAULT_CACHE_DIR = path


def cache_to_disk(file_path: Union[str, Path]) -> Callable:
    """
    A decorator to cache the result of a function to a file. If the file exists,
    the result is loaded from the file. Otherwise, the function is executed and
    the result is saved to the file.

    !!! warning "deprecated"
        This function is deprecated. Use `cache_with_joblib` instead for better
        caching capabilities including automatic cache invalidation, better object
        handling, and memory efficiency.

    ## Example usage

    ```python
    @cache_to_disk("path_to_file.pkl")
    def some_function(*args: Any, **kwargs: Any) -> Any:
        # Function implementation
        return "some result"
    ```

    Args:
        file_path (str): The path to the file where the result should be cached.

    Returns:
        Callable: The decorated function.
    """
    warnings.warn(
        "cache_to_disk is deprecated. Use cache_with_joblib instead for better "
        "caching capabilities including automatic cache invalidation, better object "
        "handling, and memory efficiency.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(file_path, str):
        file_path = Path(file_path)
    assert isinstance(file_path, Path)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if os.path.exists(file_path):
                log.info(
                    f"Loading cached result of {func.__name__} from {file_path}",
                    stacklevel=2,
                )
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    pickle.dump(result, f)
                return result

        return wrapper

    return decorator


def cache_with_joblib(
    cache_dir: Union[str, Path] = None,
    verbose: int = 0,
) -> Callable:
    """
    A decorator to cache the result of a function using joblib.Memory. This provides
    more advanced caching capabilities compared to cache_to_disk, including:
    - Automatic cache invalidation when function arguments change
    - Better handling of numpy arrays and other complex objects
    - Memory-efficient storage
    - Optional verbose output for cache hits/misses

    ## Example usage

    ```python
    @cache_with_joblib("./cache", verbose=1)
    def expensive_computation(x: int, y: str) -> Any:
        # Function implementation
        return complex_result

    # Or with default settings:
    @cache_with_joblib()
    def another_function(x: int) -> int:
        return x * 2
    ```

    Args:
        cache_dir (Union[str, Path]): The directory where cache files should be stored.
            If `None`, a default directory `outputs/cache` will be used.
        verbose (int): Verbosity level for joblib.Memory (0=silent, 1=basic, 2++=verbose).

    Returns:
        Callable: A decorator function that can be applied to functions.
    """

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    assert isinstance(cache_dir, Path)

    # Create the cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a Memory object for this function
    memory = Memory(location=cache_dir, verbose=verbose)

    def decorator(func: Callable) -> Callable:
        nonlocal memory

        # Create the cached version of the function
        cached_func = memory.cache(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cached_func(*args, **kwargs)

        # Expose useful methods from joblib.Memory
        if not (
            hasattr(cached_func, "clear")
            or hasattr(cached_func, "call")
            or hasattr(cached_func, "check_call_in_cache")
        ):
            wrapper.clear = cached_func.clear
            wrapper.call = cached_func.call
            wrapper.check_call_in_cache = cached_func.check_call_in_cache

        return wrapper

    return decorator
