from rich.live import Live


class RichLiveMixin:
    """
    A mixin class that provides Rich Live display capabilities.

    This mixin integrates Rich's Live display functionality, allowing for
    dynamic, auto-refreshing console output. It's particularly useful for
    displaying real-time updates, progress information, or continuously
    changing data without cluttering the terminal.

    Attributes:
        _rich_live (Live): The internal Rich Live instance for live display updates.

    Example:
        ```python
        class MyTask(RichLiveMixin):
            def run(self):
                self.start_rich_live()
                for i in range(100):
                    self.rich_live_print(f"Processing item {i}")
                    time.sleep(0.1)
                self.stop_rich_live()
        ```
    """

    _rich_live: Live = None

    @property
    def rich_live(self) -> Live:
        """
        Get the Rich Live instance, creating it if necessary.

        Returns:
            Live: The Rich Live instance for dynamic console output.
        """
        if self._rich_live is None:
            self._rich_live = Live()
        return self._rich_live

    def start_rich_live(self):
        """
        Start the Rich Live display context.

        This method enters the Rich Live context, enabling dynamic console output.
        Must be paired with stop_rich_live() to properly clean up resources.

        Returns:
            The Rich Live instance in its started state.

        Example:
            ```python
            self.start_rich_live()
            # Display dynamic content
            self.rich_live_print("Dynamic output")
            self.stop_rich_live()
            ```
        """
        return self.rich_live.__enter__()

    def stop_rich_live(self):
        """
        Stop the Rich Live display context and clean up resources.

        This method exits the Rich Live context and resets the internal Live instance.
        Should be called after start_rich_live() when dynamic display is complete.

        Example:
            ```python
            self.start_rich_live()
            # ... display content ...
            self.stop_rich_live()
            ```
        """
        self.rich_live.__exit__(None, None, None)
        self._rich_live = None

    def rich_live_print(self, msg):
        """
        Print a message to the Rich Live console.

        This method displays the given message through the Rich Live console,
        allowing for formatted, dynamic output that updates in place.

        Args:
            msg: The message to display. Can be a string or any Rich renderable object.

        Example:
            ```python
            self.start_rich_live()
            self.rich_live_print("[bold green]Success![/bold green]")
            self.rich_live_print(Panel("Status: Running"))
            self.stop_rich_live()
            ```
        """
        self.rich_live.console.print(msg)
