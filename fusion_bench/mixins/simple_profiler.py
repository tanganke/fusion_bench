from contextlib import contextmanager
from typing import Generator, Optional

from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.profilers import SimpleProfiler

__all__ = ["SimpleProfilerMixin"]


class SimpleProfilerMixin:
    """
    A mixin class that provides simple profiling capabilities using Lightning's SimpleProfiler.

    This mixin allows for easy profiling of code blocks using a context manager or manual
    start/stop methods. It measures the execution time of named actions and provides
    a summary of the profiling results. Unlike statistical profilers, this provides
    precise timing measurements for specific code blocks.

    Note:
        This mixin uses Lightning's SimpleProfiler which measures wall-clock time
        for named actions. It's suitable for timing discrete operations rather than
        detailed function-level profiling.

    Examples:
        ```python
        class MyClass(SimpleProfilerMixin):
            def do_something(self):
                with self.profile("data_loading"):
                    # Load data here
                    data = load_data()

                with self.profile("model_training"):
                    # Train model here
                    model.train(data)

                # Print the profiling summary
                self.print_profile_summary("Training Profile")
        ```

    Attributes:
        _profiler (SimpleProfiler): An instance of the SimpleProfiler class used for profiling.
    """

    _profiler: SimpleProfiler = None

    @property
    def profiler(self) -> SimpleProfiler:
        """
        Get the SimpleProfiler instance, creating it if necessary.

        Returns:
            SimpleProfiler: The profiler instance used for timing measurements.
        """
        # Lazy initialization of the profiler instance
        if self._profiler is None:
            self._profiler = SimpleProfiler()
        return self._profiler

    @contextmanager
    def profile(self, action_name: str) -> Generator:
        """
        Context manager for profiling a code block.

        This context manager automatically starts profiling when entering the block
        and stops profiling when exiting the block (even if an exception occurs).

        Args:
            action_name: A descriptive name for the action being profiled.
                        This name will appear in the profiling summary.

        Yields:
            str: The action name that was provided.

        Example:

        ```python
        with self.profile("data_processing"):
            # Process data here
            result = process_large_dataset()
        ```
        """
        try:
            self.start_profile(action_name)
            yield action_name
        finally:
            self.stop_profile(action_name)

    def start_profile(self, action_name: str):
        """
        Start profiling for a named action.

        This method begins timing for the specified action. You must call
        stop_profile() with the same action name to complete the measurement.

        Args:
            action_name: A descriptive name for the action being profiled.
                        This name will appear in the profiling summary.

        Example:
            ```python
            self.start_profile("model_inference")
            result = model.predict(data)
            self.stop_profile("model_inference")
            ```
        """
        self.profiler.start(action_name)

    def stop_profile(self, action_name: str):
        """
        Stop profiling for a named action.

        This method ends timing for the specified action that was previously
        started with start_profile().

        Args:
            action_name: The name of the action to stop profiling.
                        Must match the name used in start_profile().

        Example:
            ```python
            self.start_profile("data_loading")
            data = load_data()
            self.stop_profile("data_loading")
            ```
        """
        self.profiler.stop(action_name)

    @rank_zero_only
    def print_profile_summary(self, title: Optional[str] = None):
        """
        Print a summary of all profiled actions.

        This method outputs a formatted summary showing the timing information
        for all actions that have been profiled. The output includes action names
        and their execution times.

        Args:
            title: Optional title to print before the profiling summary.
                  If provided, this will be printed as a header.

        Note:
            This method is decorated with @rank_zero_only, meaning it will only
            execute on the main process in distributed training scenarios.

        Example:
            ```python
            # After profiling some actions
            self.print_profile_summary("Training Performance Summary")
            ```
        """
        if title is not None:
            print(title)
        print(self.profiler.summary())

    def __del__(self):
        """
        Cleanup when the object is destroyed.

        Ensures that the profiler instance is properly cleaned up to prevent
        memory leaks when the mixin instance is garbage collected.
        """
        if self._profiler is not None:
            del self._profiler
            self._profiler = None
