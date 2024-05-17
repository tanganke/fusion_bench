import logging
import time

log = logging.getLogger(__name__)


class timeit_context:
    """
    Usage:

    ```python
    with timeit_context() as timer:
        ... # code block to be measured
    ```
    """

    nest_level = -1

    def _log(self, msg):
        log.log(self.loglevel, msg, stacklevel=3)

    def __init__(self, msg: str = None, loglevel=logging.INFO) -> None:
        self.loglevel = loglevel
        self.msg = msg

    def __enter__(self) -> None:
        """
        Sets the start time and logs an optional message indicating the start of the code block execution.

        Args:
            msg: str, optional message to log
        """
        self.start_time = time.time()
        timeit_context.nest_level += 1
        if self.msg is not None:
            self._log("  " * timeit_context.nest_level + "[BEGIN] " + str(self.msg))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Calculates the elapsed time and logs it, along with an optional message indicating the end of the code block execution.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self._log(
            "  " * timeit_context.nest_level
            + "[END]   "
            + str(f"Elapsed time: {elapsed_time:.2f}s")
        )
        timeit_context.nest_level -= 1
