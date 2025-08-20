import threading
from pathlib import Path
from typing import Optional, Union


class RuntimeConstants:
    """
    This class holds constants related to the runtime environment of the Fusion Bench framework.
    It includes default values for cache directories and other runtime configurations.

    Implemented as a thread-safe singleton to ensure consistent runtime configuration
    across the entire application.
    """

    _instance: Optional["RuntimeConstants"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RuntimeConstants":
        """Create a new instance using singleton pattern with thread safety."""
        with cls._lock:
            # Double-check locking pattern
            if cls._instance is None:
                cls._instance = super(RuntimeConstants, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the singleton instance only once."""
        if not self._initialized:
            # Add your runtime constants here
            self._initialized = True

    debug = False

    @property
    def cache_dir(self) -> Path:
        from fusion_bench.utils.cache_utils import DEFAULT_CACHE_DIR

        return DEFAULT_CACHE_DIR

    @cache_dir.setter
    def cache_dir(self, path: Union[str, Path]) -> None:
        from fusion_bench.utils.cache_utils import set_default_cache_dir

        set_default_cache_dir(path)

    @property
    def print_function_call(self) -> bool:
        from fusion_bench.utils.instantiate_utils import PRINT_FUNCTION_CALL

        return PRINT_FUNCTION_CALL

    @print_function_call.setter
    def print_function_call(self, enable: bool) -> None:
        from fusion_bench.utils.instantiate_utils import set_print_function_call

        set_print_function_call(enable)
