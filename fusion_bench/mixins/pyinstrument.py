from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from lightning.fabric.utilities.rank_zero import rank_zero_only
from pyinstrument import Profiler

__all__ = ["PyinstrumentProfilerMixin"]


class PyinstrumentProfilerMixin:
    """
    A mixin class that provides statistical profiling capabilities using pyinstrument.

    This mixin allows for easy profiling of code blocks using a context manager.
    It provides methods to start and stop profiling actions, save profiling results
    to files, and print profiling summaries.

    Note:
        This mixin requires the `pyinstrument` package to be installed.
        If not available, an ImportError will be raised when importing this module.

    Examples:

    ```python
    class MyClass(PyinstrumentProfilerMixin):
        def do_something(self):
            with self.profile("work"):
                # do some work here
                ...

            # save the profiling results
            self.save_profile_report("profile_report.html")

            # or print the summary
            self.print_profile_summary()
    ```

    Attributes:
        _profiler (Profiler): An instance of the pyinstrument Profiler class.
    """

    _profiler: Optional[Profiler] = None
    _is_profiling: bool = False

    @property
    def profiler(self) -> Optional[Profiler]:
        """Get the profiler instance, creating it if necessary."""
        if self._profiler is None:
            self._profiler = Profiler()
        return self._profiler

    @contextmanager
    def profile(self, action_name: Optional[str] = None) -> Generator:
        """
        Context manager for profiling a code block.

        Args:
            action_name: Optional name for the profiling action (for logging purposes).

        Example:

        ```python
        with self.profile("expensive_operation"):
            # do some expensive work here
            expensive_function()
        ```
        """
        try:
            self.start_profile(action_name)
            yield action_name
        finally:
            self.stop_profile(action_name)

    def start_profile(self, action_name: Optional[str] = None):
        """
        Start profiling.

        Args:
            action_name: Optional name for the profiling action.
        """
        if self._is_profiling:
            return

        self.profiler.start()
        self._is_profiling = True
        if action_name:
            print(f"Started profiling: {action_name}")

    def stop_profile(self, action_name: Optional[str] = None):
        """
        Stop profiling.

        Args:
            action_name: Optional name for the profiling action.
        """
        if not self._is_profiling:
            return

        self.profiler.stop()
        self._is_profiling = False
        if action_name:
            print(f"Stopped profiling: {action_name}")

    @rank_zero_only
    def print_profile_summary(
        self, title: Optional[str] = None, unicode: bool = True, color: bool = True
    ):
        """
        Print a summary of the profiling results.

        Args:
            title: Optional title to print before the summary.
            unicode: Whether to use unicode characters in the output.
            color: Whether to use color in the output.
        """
        if self.profiler is None:
            print("No profiling data available.")
            return

        if title is not None:
            print(title)

        print(self.profiler.output_text(unicode=unicode, color=color))

    @rank_zero_only
    def save_profile_report(
        self,
        output_path: Union[str, Path] = "profile_report.html",
        format: str = "html",
        title: Optional[str] = None,
    ):
        """
        Save the profiling results to a file.

        Args:
            output_path: Path where to save the profiling report.
            format: Output format ('html', or 'text').
            title: Optional title for the report.
        """
        if self.profiler is None:
            print("No profiling data available.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "html":
            content = self.profiler.output_html()
        elif format.lower() == "text":
            content = self.profiler.output_text(unicode=True, color=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', or 'text'.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Profile report saved to: {output_path}")

    def reset_profile(self):
        """Reset the profiler to start fresh."""
        if self._is_profiling:
            self.stop_profile()

        self._profiler = None

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if self._is_profiling:
            self.stop_profile()

        if self._profiler is not None:
            del self._profiler
            self._profiler = None
