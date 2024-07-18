from contextlib import contextmanager
from typing import Generator

from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.profilers import SimpleProfiler


class SimpleProfilerMixin:
    _profiler: SimpleProfiler = None

    @property
    def profiler(self):
        if self._profiler is None:
            self._profiler = SimpleProfiler()
        return self._profiler

    @contextmanager
    def profile(self, action_name: str) -> Generator:
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
    def print_profile_summary(self):
        print(self.profiler.summary())

    def __del__(self):
        if self._profiler is not None:
            self._profiler.summary()
            del self._profiler
            self._profiler = None
