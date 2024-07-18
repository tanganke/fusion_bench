from rich.live import Live


class RichLiveMixin:
    _rich_live: Live = None

    @property
    def rich_live(self) -> Live:
        if self._rich_live is None:
            self._rich_live = Live()
        return self._rich_live

    def start_rich_live(self):
        return self.rich_live.__enter__()

    def stop_rich_live(self):
        self.rich_live.__exit__(None, None, None)
        self._rich_live = None

    def rich_live_print(self, msg):
        self.rich_live.console.print(msg)
