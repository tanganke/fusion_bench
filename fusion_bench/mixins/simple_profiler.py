from lightning.pytorch.profilers import SimpleProfiler


class SimpleProfilerMixin:
    _profiler: SimpleProfiler = None

    @property
    def profiler(self):
        if self._profiler is None:
            self._profiler = SimpleProfiler()
        return self._profiler

    def start_profiler(self, action_name: str):
        self.profiler.start(action_name)

    def end_profiler(self, action_name: str):
        self.profiler.stop(action_name)

    def __del__(self):
        if self._profiler is not None:
            self._profiler.summary()
            del self._profiler
            self._profiler = None
