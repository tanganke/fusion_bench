import logging
import time

log = logging.getLogger(__name__)


class timeit_context:
    """
    A context manager for measuring and logging execution time of code blocks.

    This context manager provides precise timing measurements with automatic logging
    of elapsed time. It supports nested timing contexts with proper indentation
    for hierarchical timing analysis, making it ideal for profiling complex
    operations with multiple sub-components.

    Args:
        msg (str, optional): Custom message to identify the timed code block.
            If provided, logs "[BEGIN] {msg}" at start and includes context
            in the final timing report. Defaults to None.
        loglevel (int, optional): Python logging level for output messages.
            Uses standard logging levels (DEBUG=10, INFO=20, WARNING=30, etc.).
            Defaults to logging.INFO.

    Example:
        Basic usage:
        ```python
        with timeit_context("data loading"):
            data = load_large_dataset()
        # Logs: [BEGIN] data loading
        # Logs: [END]   Elapsed time: 2.34s
        ```

        Nested timing:
        ```python
        with timeit_context("model training"):
            with timeit_context("data preprocessing"):
                preprocess_data()
            with timeit_context("forward pass"):
                model(data)
        # Output shows nested structure:
        # [BEGIN] model training
        #   [BEGIN] data preprocessing
        #   [END]   Elapsed time: 0.15s
        #   [BEGIN] forward pass
        #   [END]   Elapsed time: 0.89s
        # [END]   Elapsed time: 1.04s
        ```

        Custom log level:
        ```python
        with timeit_context("debug operation", loglevel=logging.DEBUG):
            debug_function()
        ```
    """

    nest_level = -1

    def _log(self, msg):
        """
        Internal method for logging messages with appropriate stack level.

        This helper method ensures that log messages appear to originate from
        the caller's code rather than from internal timer methods, providing
        more useful debugging information.

        Args:
            msg (str): The message to log at the configured log level.
        """
        log.log(self.loglevel, msg, stacklevel=3)

    def __init__(self, msg: str = None, loglevel=logging.INFO) -> None:
        """
        Initialize a new timing context with optional message and log level.

        Args:
            msg (str, optional): Descriptive message for the timed operation.
                If provided, will be included in the begin/end log messages
                to help identify what is being timed. Defaults to None.
            loglevel (int, optional): Python logging level for timer output.
                Common values include:
                - logging.DEBUG (10): Detailed debugging information
                - logging.INFO (20): General information (default)
                - logging.WARNING (30): Warning messages
                - logging.ERROR (40): Error messages
                Defaults to logging.INFO.
        """
        self.loglevel = loglevel
        self.msg = msg

    def __enter__(self) -> None:
        """
        Enter the timing context and start the timer.

        This method is automatically called when entering the 'with' statement.
        It records the current timestamp, increments the nesting level for
        proper log indentation, and optionally logs a begin message.

        Returns:
            None: This context manager doesn't return a value to the 'as' clause.
                  All timing information is handled internally and logged automatically.
        """
        self.start_time = time.time()
        timeit_context.nest_level += 1
        if self.msg is not None:
            self._log("  " * timeit_context.nest_level + "[BEGIN] " + str(self.msg))

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the timing context and log the elapsed time.

        This method is automatically called when exiting the 'with' statement,
        whether through normal completion or exception. It calculates the total
        elapsed time and logs the results with proper nesting indentation.

        Args:
            exc_type (type): Exception type if an exception occurred, None otherwise.
            exc_val (Exception): Exception instance if an exception occurred, None otherwise.
            exc_tb (traceback): Exception traceback if an exception occurred, None otherwise.

        Returns:
            None: Does not suppress exceptions (returns None/False implicitly).
                  Any exceptions that occurred in the timed block will propagate normally.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self._log(
            "  " * timeit_context.nest_level
            + "[END]   "
            + str(f"Elapsed time: {elapsed_time:.2f}s")
        )
        timeit_context.nest_level -= 1
