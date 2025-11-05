"""
Runtime Constants Module.

This module provides a thread-safe singleton class for managing runtime configuration
and constants across the Fusion Bench framework. It centralizes access to runtime
settings like cache directories, debug flags, and logging preferences.

Example:
    ```python
    from fusion_bench.constants.runtime import RuntimeConstants

    # Get the singleton instance
    runtime = RuntimeConstants()

    # Configure cache directory
    runtime.cache_dir = "/custom/cache/path"

    # Enable debug mode
    runtime.debug = True

    # Control function call logging
    runtime.print_function_call = True
    ```
"""

import os
import threading
from pathlib import Path
from typing import Optional, Union


class RuntimeConstants:
    """
    Thread-safe singleton for managing runtime configuration and constants.

    This class provides centralized access to runtime settings that affect the
    behavior of the entire Fusion Bench framework. It ensures consistent
    configuration across all modules and supports thread-safe access in
    multi-threaded environments.

    Attributes:
        debug: Global debug flag for enabling verbose logging and debugging features.

    Example:
        ```python
        runtime = RuntimeConstants()

        # Configure caching
        runtime.cache_dir = Path.home() / ".cache" / "fusion_bench"

        # Enable debugging
        runtime.debug = True
        runtime.print_function_call = True
        ```

    Note:
        This class implements the singleton pattern with thread-safe initialization.
        Multiple calls to the constructor will return the same instance.
    """

    _instance: Optional["RuntimeConstants"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RuntimeConstants":
        """
        Create or return the singleton instance with thread safety.

        Uses double-check locking pattern to ensure thread-safe singleton creation
        while minimizing synchronization overhead.

        Returns:
            The singleton RuntimeConstants instance.
        """
        with cls._lock:
            # Double-check locking pattern
            if cls._instance is None:
                cls._instance = super(RuntimeConstants, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """
        Initialize the singleton instance only once.

        Subsequent calls to __init__ are no-ops to maintain singleton behavior.
        """
        if not self._initialized:
            # Initialize default values
            self._initialized = True

    debug = False
    """Global debug flag for enabling verbose logging and debugging features."""

    @property
    def cache_dir(self) -> Path:
        """
        Get the default cache directory for models and datasets.

        Returns:
            Path object pointing to the cache directory.

        Example:
            ```python
            runtime = RuntimeConstants()
            print(f"Cache directory: {runtime.cache_dir}")
            ```
        """
        from fusion_bench.utils.cache_utils import DEFAULT_CACHE_DIR

        return DEFAULT_CACHE_DIR

    @cache_dir.setter
    def cache_dir(self, path: Union[str, Path]) -> None:
        """
        Set the default cache directory for models and datasets.

        Args:
            path: New cache directory path as string or Path object.

        Example:
            ```python
            runtime = RuntimeConstants()
            runtime.cache_dir = "/data/fusion_bench_cache"
            ```
        """
        from fusion_bench.utils.cache_utils import set_default_cache_dir

        set_default_cache_dir(path)

    @property
    def print_function_call(self) -> bool:
        """
        Get whether function calls are printed during instantiation.

        Returns:
            True if function call printing is enabled, False otherwise.
        """
        from fusion_bench.utils.instantiate_utils import PRINT_FUNCTION_CALL

        return PRINT_FUNCTION_CALL

    @print_function_call.setter
    def print_function_call(self, enable: bool) -> None:
        """
        Set whether to print function calls during instantiation.

        Useful for debugging to see which functions are being called
        when instantiating objects from configuration.

        Args:
            enable: True to enable printing, False to disable.

        Example:
            ```python
            runtime = RuntimeConstants()
            runtime.print_function_call = True  # Enable verbose logging
            ```
        """
        from fusion_bench.utils.instantiate_utils import set_print_function_call

        set_print_function_call(enable)
