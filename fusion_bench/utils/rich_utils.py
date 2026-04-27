import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

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

# Syntax highlighting themes optimized for dark and light terminals
_DARK_THEME = "monokai"
_LIGHT_THEME = "solarized-light"

# Cache the detection result (avoids repeated TTY probes)
_terminal_theme_cache: Optional[str] = None


def _detect_terminal_theme() -> str:
    """Detect whether the terminal uses a light or dark background.

    Returns ``"light"`` or ``"dark"``.  Detection order:
    1. Environment variable (``TERM_COLOR_SCHEME``, ``CLAUDE_COLOR_SCHEME``,
       ``VSCODE_COLOR_THEME``).
    2. Actual terminal background colour probed via OSC 11 query (works in
       Kitty, Alacritty, GNOME Terminal, and most modern xterm-compatible
       terminals).
    3. Fall back to ``"dark"`` for non-TTY / pipes / unknown environments.
    """
    # 1. Explicit env override
    for env_var in ("TERM_COLOR_SCHEME", "CLAUDE_COLOR_SCHEME", "VSCODE_COLOR_THEME"):
        value = os.environ.get(env_var, "").lower().strip()
        if value in ("light", "dark"):
            return value

    # 2. Probe terminal background colour via OSC 11
    #    Only attempt on real TTYs; otherwise fall back.
    if not os.isatty(sys.stdout.fileno()):
        return "dark"

    try:
        import select
        import termios
        import tty

        # Send OSC 11 query: "what is my background color?"
        query = b"\x1b]11;?\x1b\\"
        os.write(sys.stdout.fileno(), query)
        sys.stdout.flush()

        old_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setraw(sys.stdin.fileno())
            response = b""
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    break
                ch = os.read(sys.stdin.fileno(), 1)
                response += ch
                # Stop when we see the OSC 11 response marker or have enough data
                if b"\x1b]11;" in response or len(response) > 200:
                    break
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

        # Parse response. Terminals reply in various formats:
        #   OSC 11 ; #RRGGBB ST
        #   OSC 11 ; R,G,B ST  (values 0-65535)
        #   OSC 11 ; rgb:RRRR/GGGG/BBBB ST
        ESC = b"\x1b"

        # Try #RRGGBB format
        match = re.search(ESC + b"]11;#([0-9a-fA-F]{6})", response)
        if match:
            hex_color = match.group(1).decode()
            r, g, b = (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
        else:
            # Try R,G,B (0-65535) format
            match = re.search(ESC + rb"]11;([0-9]+),([0-9]+),([0-9]+)", response)
            if match:
                r, g, b = (
                    int(match.group(1)) // 257,
                    int(match.group(2)) // 257,
                    int(match.group(3)) // 257,
                )
            else:
                # Try rgb:RRRR/GGGG/BBBB (16-bit values)
                match = re.search(
                    ESC + rb"]11;rgb:([0-9a-fA-F]{4})/([0-9a-fA-F]{4})/([0-9a-fA-F]{4})", response
                )
                if match:
                    r, g, b = (
                        int(match.group(1).decode(), 16) // 257,
                        int(match.group(2).decode(), 16) // 257,
                        int(match.group(3).decode(), 16) // 257,
                    )
                else:
                    return "dark"  # couldn't parse response

    except (OSError, ImportError, ValueError):
        return "dark"

    # Luminance heuristic: average > 128 -> light
    if (r + g + b) / 3 > 128:
        return "light"
    return "dark"


def get_syntax_theme() -> Tuple[str, Optional[str]]:
    """Return ``(theme_name, background_color)`` suited to the current terminal.

    Use this instead of hard-coding ``"monokai"`` when creating
    :class:`rich.syntax.Syntax` objects.
    """
    global _terminal_theme_cache

    if _terminal_theme_cache is None:
        _terminal_theme_cache = _detect_terminal_theme()

    if _terminal_theme_cache == "light":
        return _LIGHT_THEME, "default"
    return _DARK_THEME, "default"


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
    theme: Optional[str] = None,
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
        theme (str, optional): Syntax highlighting theme. Defaults to auto-detection
            based on terminal background (dark theme for dark terminals, light theme
            for light terminals).
    """
    if theme is None:
        theme, background_color = get_syntax_theme()
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
    theme: Optional[str] = None,
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
            ``code_style`` is provided. Defaults to auto-detection based on terminal
            background.
        background_color (str, optional): Background color style to apply to the code
            block when using syntax highlighting. Defaults to ``"default"``.
        print_fn (Callable, optional): Function used to render the resulting Rich
            object. Defaults to :func:`rich.print`.
    """
    if theme is None:
        theme, background_color = get_syntax_theme()
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
    theme: Optional[str] = None,
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
        theme (str, optional): The theme to use for syntax highlighting. Defaults to
            auto-detection based on terminal background.
        background_color (str, optional): The background color to use for syntax highlighting.
            Defaults to "default".

    Returns:
        None
    """
    if theme is None:
        theme, background_color = get_syntax_theme()
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
    theme: Optional[str] = None,
    background_color: Optional[str] = "default",
) -> None:
    """
    Prints the contents of a DictConfig as a YAML string using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
        output_path: Optional path to export the config YAML to. If provided, the file is written to this path.
        theme (str, optional): The theme to use for syntax highlighting. Defaults to
            auto-detection based on terminal background.
    """
    if theme is None:
        theme, background_color = get_syntax_theme()
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
