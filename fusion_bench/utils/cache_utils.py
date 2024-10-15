import logging
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

__all__ = ["cache_to_disk"]


log = logging.getLogger(__name__)


def cache_to_disk(file_path: Union[str, Path]) -> Callable:
    """
    A decorator to cache the result of a function to a file. If the file exists,
    the result is loaded from the file. Otherwise, the function is executed and
    the result is saved to the file.

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
