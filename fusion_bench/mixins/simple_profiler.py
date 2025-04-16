from contextlib import contextmanager
from typing import Generator, Optional

from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.profilers import SimpleProfiler

__all__ = ["SimpleProfilerMixin"]


class SimpleProfilerMixin:
    """
    A mixin class that provides simple profiling capabilities.

    This mixin allows for easy profiling of code blocks using a context manager.
    It also provides methods to start and stop profiling actions, and to print
    a summary of the profiling results.

    Examples:

    ```python
    class MyClass(SimpleProfilerMixin):
        def do_something(self):
            with self.profile("work"):
                # do some work here
                ...
            with self.profile("more work"):
                # do more work here
                ...

            # print the profiling summary
            self.print_profile_summary()
    ```

    Attributes:
        _profiler (SimpleProfiler): An instance of the SimpleProfiler class used for profiling.
    """

    _profiler: SimpleProfiler = None

    @property
    def profiler(self):
        # Lazy initialization of the profiler instance
        if self._profiler is None:
            self._profiler = SimpleProfiler()
        return self._profiler

    @contextmanager
    def profile(self, action_name: str) -> Generator:
        """
        Context manager for profiling a code block

        Example:

        ```python
        with self.profile("work"):
            # do some work here
            ...
        ```
        """
        try:
            self.start_profile(action_name)
            yield action_name
        finally:
            self.stop_profile(action_name)

    def start_profile(self, action_name: str):
        self.profiler.start(action_name)

    def stop_profile(self, action_name: str):
        self.profiler.stop(action_name)

    @rank_zero_only
    def print_profile_summary(self, title: Optional[str] = None):
        if title is not None:
            print(title)
        print(self.profiler.summary())

    def __del__(self):
        if self._profiler is not None:
            del self._profiler
            self._profiler = None
