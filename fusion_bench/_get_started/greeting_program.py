import logging
from typing import Optional

from omegaconf import DictConfig

from fusion_bench.programs import BaseHydraProgram

log = logging.getLogger(__name__)


class GreetingProgram(BaseHydraProgram):
    """
    A simple program that greets users with a custom message.
    """

    _config_mapping = BaseHydraProgram._config_mapping | {
        "message": "message",
        "name": "name",
        "repeat_count": "repeat_count",
    }

    def __init__(
        self,
        message: str = "Hello",
        name: str = "World",
        repeat_count: int = 1,
        **kwargs,
    ):
        self.message = message
        self.name = name
        self.repeat_count = repeat_count
        super().__init__(**kwargs)

    def run(self):
        """Execute the greeting workflow."""
        log.info("Starting greeting program")

        # Create the greeting
        greeting = f"{self.message}, {self.name}!"

        # Print the greeting multiple times
        for i in range(self.repeat_count):
            if self.repeat_count > 1:
                print(f"[{i+1}/{self.repeat_count}] {greeting}")
            else:
                print(greeting)

        log.info("Greeting program completed")
        return greeting
