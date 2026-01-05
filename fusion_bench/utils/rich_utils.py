import logging
from pathlib import Path
from typing import Optional, Sequence

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
from rich.traceback import install as install_rich_traceback

from fusion_bench.utils import pylogger
from fusion_bench.utils.packages import _is_package_available

install_rich_traceback()

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


def format_code_str(message: str, code_style="python"):
    if code_style.lower() == "python" and _is_package_available("black"):
        # Use black formatting for python code if black is available
        import black

        try:
            message = black.format_str(message, mode=black.Mode())
        except black.InvalidInput:
            pass  # If black fails, use the original message

    return message.strip()


def print_bordered(
    message,
    title=None,
    style="blue",
    code_style=None,
    *,
    expand: bool = True,
    theme: str = "monokai",
    background_color: Optional[str] = "default",
    print_fn=print,
    format_code: bool = True,
):
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
        if format_code:
            message = format_code_str(message, code_style)
        content = Syntax(
            message,
            code_style,
            word_wrap=True,
            theme=theme,
            background_color=background_color,
        )
    else:
        content = Text(message)

    panel = Panel(content, title=title, border_style=style, expand=expand)
    print_fn(panel)


def print_code(
    message,
    title=None,
    code_style=None,
    *,
    expand: bool = True,
    theme: str = "monokai",
    background_color: Optional[str] = "default",
    print_fn=print,
):
    """
    Print code or plain text with optional syntax highlighting.

    Args:
        message (str): The message or code to print.
        title (str, optional): Optional title associated with this output. Currently
            not used by this function, but kept for API compatibility. Defaults to None.
        code_style (str, optional): The language/lexer name for syntax highlighting
            (for example, ``"python"``). If ``None``, the message is rendered as plain
            text without syntax highlighting. Defaults to ``None``.
        expand (bool, optional): Placeholder flag for API symmetry with other printing
            helpers. It is not used in the current implementation. Defaults to True.
        theme (str, optional): Name of the Rich syntax highlighting theme to use when
            ``code_style`` is provided. Defaults to ``"monokai"``.
        background_color (str, optional): Background color style to apply to the code
            block when using syntax highlighting. Defaults to ``"default"``.
        print_fn (Callable, optional): Function used to render the resulting Rich
            object. Defaults to :func:`rich.print`.
    """
    if code_style:
        content = Syntax(
            message,
            code_style,
            word_wrap=True,
            theme=theme,
            background_color=background_color,
        )
    else:
        content = Text(message)

    print_fn(content)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "path",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
    *,
    theme: str = "monokai",
    background_color: Optional[str] = "default",
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg (DictConfig): A DictConfig composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
            Defaults to ``("data", "model", "callbacks", "logger", "trainer", "paths", "extras")``.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
            Defaults to ``False``.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
            Defaults to ``False``.
        theme (str, optional): The theme to use for syntax highlighting. Defaults to "monokai".
        background_color (str, optional): The background color to use for syntax highlighting.
            Defaults to "default".

    Returns:
        None
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

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve).strip()
        else:
            branch_content = str(config_group)

        branch.add(
            rich.syntax.Syntax(
                branch_content,
                "yaml",
                theme=theme,
                background_color=background_color,
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    other_fields = [field for field in cfg if field not in queue]
    if other_fields:
        others_branch = tree.add(Text("[others]"), style=style, guide_style=style)

        other_cfg = OmegaConf.create({field: cfg[field] for field in other_fields})
        branch_content = OmegaConf.to_yaml(other_cfg, resolve=resolve).strip()

        others_branch.add(
            rich.syntax.Syntax(
                branch_content, "yaml", theme=theme, background_color=background_color
            )
        )

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        if not cfg.get("paths") or not cfg.paths.get("output_dir"):
            log.error(
                "Cannot save config tree to file. 'paths.output_dir' is not specified in the config."
            )
        else:
            with open(Path(cfg.path.output_dir, "config_tree.log"), "w") as file:
                rich.print(tree, file=file)


@rank_zero_only
def print_config_yaml(
    cfg: DictConfig,
    resolve: bool = False,
    output_path: Optional[str] = False,
    *,
    theme: str = "monokai",
    background_color: Optional[str] = "default",
) -> None:
    """
    Prints the contents of a DictConfig as a YAML string using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
        output_path: Optional path to export the config YAML to. If provided, the file is written to this path.
    """
    config_yaml = OmegaConf.to_yaml(cfg, resolve=resolve)
    syntax = rich.syntax.Syntax(
        config_yaml, "yaml", theme=theme, background_color=background_color
    )
    rich.print(syntax)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(output_path), "w") as file:
            rich.print(syntax, file=file)


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


def setup_colorlogging(
    force=False,
    level=logging.INFO,
    **kwargs,
):
    """
    Sets up color logging for the application.
    """
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
        force=force,
        **kwargs,
    )
