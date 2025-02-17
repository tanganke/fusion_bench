import logging
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from fusion_bench.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# List of available styles
AVAILABLE_STYLES = [
    "none",
    "bold",
    "dim",
    "italic",
    "underline",
    "blink",
    "blink2",
    "reverse",
    "conceal",
    "strike",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
]


def display_available_styles():
    """Display all available styles in a grid."""
    console = Console()
    style_samples = [
        Panel(f"Style: {style}", expand=False, border_style=style)
        for style in AVAILABLE_STYLES
    ]
    console.print(Columns(style_samples, equal=True, expand=False))


def print_bordered(message, title=None, style="blue", code_style=None):
    """
    Print a message with a colored border.

    Args:
    message (str): The message to print.
    title (str, optional): The title of the panel. Defaults to None.
    style (str, optional): The color style for the border. Defaults to "cyan".
    code_style (str, optional): The syntax highlighting style if the message is code.
                                Set to None for plain text. Defaults to "python".
    """
    if code_style:
        content = Syntax(message, code_style, theme="monokai", word_wrap=True)
    else:
        content = Text(message)

    panel = Panel(content, title=title, border_style=style)
    print(panel)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "tree"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


if __name__ == "__main__":
    print_bordered("Hello, World!", title="Greeting", style="green", code_style=None)
    print_bordered(
        "def hello_world():\n    print('Hello, World!')",
        title="Python Function",
        style="magenta",
        code_style="python",
    )
    print_bordered(
        "SELECT * FROM users WHERE age > 18;",
        title="SQL Query",
        style="yellow",
        code_style="sql",
    )

    print("\nAvailable Styles:")
    display_available_styles()


def setup_colorlogging(force=False, **config_kwargs):
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
        force=force,
        **config_kwargs,
    )
