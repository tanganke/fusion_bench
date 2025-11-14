"""
Web UI for FusionBench Command Generator with per-session state management.
"""

import argparse
import functools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import gradio as gr
import hydra
import yaml
from colorama import Fore, Style  # For cross-platform color support
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from fusion_bench.scripts.cli import _get_default_config_path


def escape_overrides(value: str) -> str:
    """
    Escapes special characters in Hydra command-line override values.

    Adds quotes around values containing spaces and escapes equals signs
    to prevent them from being interpreted as key-value separators.

    Args:
        value (str): The override value to escape.

    Returns:
        str: The escaped value ready for use in command-line overrides.
    """
    if " " in value and not (value.startswith('"') or value.startswith("'")):
        return f"'{value}'"
    if "=" in value:
        return value.replace("=", "\\=")
    return value


class ConfigGroupNode:
    """
    Represents a node in the configuration directory tree.

    This class recursively builds a tree structure representing the Hydra
    configuration directory hierarchy, including subdirectories (child groups)
    and YAML configuration files.

    Attributes:
        name (str): Name of the configuration group (directory name).
        path (Path): Full path to the directory.
        parent (Optional[ConfigGroupNode]): Parent node in the tree.
        children (List[ConfigGroupNode]): Child directory nodes.
        configs (List[str]): List of YAML config file names (without extension).
    """

    name: str
    path: Path
    parent: Optional["ConfigGroupNode"]
    children: List["ConfigGroupNode"]
    configs: List[str]

    def __init__(self, path: str | Path, parent: Optional["ConfigGroupNode"] = None):
        """
        Initialize a ConfigGroupNode.

        Args:
            path: Path to the configuration directory.
            parent: Parent node in the tree (None for root).
        """
        self.path = Path(path)
        assert self.path.is_dir()
        self.name = self.path.stem
        self.parent = parent
        self.children = []
        self.configs = []
        for child in self.path.iterdir():
            if child.is_dir():
                child_node = ConfigGroupNode(child, parent=self)
                self.children.append(child_node)
            elif child.is_file() and child.suffix == ".yaml":
                self.configs.append(child.stem)

    def __repr__(self):
        """
        Return a colored string representation of the tree structure.

        Returns:
            str: Tree structure with colored group names.
        """
        return f"{Fore.BLUE}{self.name}{Style.RESET_ALL}\n" + self._repr_indented()

    def _repr_indented(self, prefix=""):
        """
        Generate indented tree representation recursively.

        Args:
            prefix: String prefix for indentation.

        Returns:
            str: Indented tree structure.
        """
        result = ""

        items = self.configs + self.children
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = prefix + ("└── " if is_last else "├── ")
            next_prefix = prefix + ("    " if is_last else "│   ")

            if isinstance(item, str):  # It's a config file
                result += f"{current_prefix}{item}\n"
            else:  # It's a child directory
                result += f"{current_prefix}{Fore.BLUE}{item.name}{Style.RESET_ALL}\n"
                result += item._repr_indented(next_prefix)

        return result

    def has_child_group(self, name: str) -> bool:
        """
        Check if this node has a child group with the given name.

        Args:
            name: Name of the child group to check.

        Returns:
            bool: True if child group exists, False otherwise.
        """
        return any(child.name == name for child in self.children)

    def __getitem__(self, key: str) -> Union["ConfigGroupNode", str]:
        """
        Get a child group or config by name.

        Args:
            key: Name of the child group or config file.

        Returns:
            Union[ConfigGroupNode, str]: Child node or config file name.

        Raises:
            KeyError: If no child group or config with that name exists.
        """
        for child in self.children:
            if child.name == key:
                return child
        for config in self.configs:
            if config == key:
                return config
        raise KeyError(f"No child group or config named {key}")

    @functools.cached_property
    def prefix(self) -> str:
        """
        Get the dot-separated prefix path from root to this node.

        Returns:
            str: Prefix path (e.g., "method.modelpool.").
        """
        if self.parent is None:
            return ""
        return self.parent.prefix + self.name + "."


def priority_iterable(iter, priority_keys):
    """
    Iterate over items with priority keys first, then remaining items.

    Args:
        iter: Iterable to process.
        priority_keys: Keys to yield first.

    Yields:
        Items from iter, with priority_keys first.
    """
    items = list(iter)
    for key in priority_keys:
        if key in items:
            yield key
            items.remove(key)
    for item in items:
        yield item


class AppState:
    """
    Per-session state of the app.

    Manages the current configuration state including the selected config name,
    overrides, and the composed Hydra configuration.

    Attributes:
        config_name (str): Name of the root configuration file.
        hydra_options (List[str]): Hydra-specific command line options.
        overrides (List[str]): List of configuration overrides.
        config (DictConfig): The composed OmegaConf configuration.
    """

    config_name: str
    hydra_options: List[str]
    overrides: List[str]
    config: DictConfig

    def __init__(
        self,
        config_path: str,
        config_name: str,
        hydra_options: List[str] = [],
        overrides: List[str] = [],
    ) -> None:
        """
        Initialize the application state.

        Args:
            config_path: Path to the config directory.
            config_name: Name of the root config file.
            hydra_options: Hydra command line options.
            overrides: Initial configuration overrides.
        """
        super().__init__()
        self.config_path = config_path
        self.config_name = config_name
        self.hydra_options = hydra_options
        self.overrides = overrides
        self.update_config(config_name)

    @property
    def config_str(self):
        """
        Get the YAML string representation of the current configuration.

        Returns:
            str: YAML formatted configuration.
        """
        return OmegaConf.to_yaml(self.config)

    def update_config(
        self,
        config_name: str = None,
        overrides: List[str] = None,
    ) -> "AppState":
        """
        Update the configuration with new name and/or overrides.

        Args:
            config_name: New root config name (optional).
            overrides: New list of overrides (optional).

        Returns:
            AppState: Self for method chaining.
        """
        if config_name is not None:
            self.config_name = config_name
        if overrides is not None:
            self.overrides = overrides

        if self.config_name is None:
            self.config = ""
        else:
            self.config = compose(
                config_name=self.config_name,
                overrides=self.overrides,
                return_hydra_config=True,
            )
            HydraConfig().set_config(self.config)
            del self.config.hydra
        return self

    def generate_command(self):
        """
        Generate the fusion_bench CLI command from current state.

        Returns:
            str: Complete command ready to execute in shell.
        """
        # Generate the command according to `config_name` and `overrides` (a list of strings)
        command = "fusion_bench \\\n"
        if self.config_path is not None:
            command += f"--config-path {self.config_path} \\\n"
        command += f"--config-name {self.config_name} \\\n"
        command += " \\\n".join(self.overrides)
        command = command.strip()
        command = command.strip("\\")
        command = command.strip()

        return command

    @property
    def config_str_and_command(self):
        """
        Get both config string and command as a tuple.

        Returns:
            Tuple[str, str]: (YAML config, shell command).
        """
        return self.config_str, self.generate_command()

    def get_override(self, key: str):
        """
        Get the override value for a specific key.

        Args:
            key: Configuration key to look up.

        Returns:
            Optional[str]: Override value or None if not found.
        """
        for ov in self.overrides:
            if ov.startswith(f"{key}="):
                return "".join(ov.split("=")[1:])
        return None

    def update_override(self, key: str, value):
        """
        Update or add an override for a specific key.

        Args:
            key: Configuration key to override.
            value: New value for the key.

        Returns:
            AppState: Updated state after recomposing config.
        """
        self.overrides = [ov for ov in self.overrides if not ov.startswith(f"{key}=")]
        if value:
            self.overrides.append(f"{key}={escape_overrides(value)}")
        return self.update_config()


class App:
    """
    Main application class for the FusionBench WebUI.

    Manages the Gradio interface, configuration tree, and application state.

    Attributes:
        args: Command line arguments.
        group_tree (ConfigGroupNode): Root of the config directory tree.
        init_config_name (str): Initial configuration name.
        app_state (AppState): Current application state.
    """

    def __init__(self, args):
        """
        Initialize the application.

        Args:
            args: Parsed command line arguments.
        """
        super().__init__()
        self.args = args
        group_tree = ConfigGroupNode(self.config_path)
        if args.print_tree:
            print(group_tree)

        self.group_tree = group_tree

        if "fabric_model_fusion" in group_tree.configs:
            self.init_config_name = "fabric_model_fusion"
        else:
            self.init_config_name = group_tree.configs[0]

        initialize_config_dir(
            config_dir=self.config_path,
            job_name=Path(__file__).stem,
            version_base=None,
        )

        self.app_state = AppState(
            config_path=args.config_path,
            config_name=self.init_config_name,
            hydra_options=[],
            overrides=[],
        )

    @functools.cached_property
    def config_path(self):
        """
        Get the configuration directory path.

        Returns:
            Path: Path to the config directory.
        """
        if self.args.config_path:
            return Path(self.args.config_path)
        else:
            return _get_default_config_path()

    def __getattr__(self, name):
        """
        Delegate attribute access to app_state if not found in App.

        Args:
            name: Attribute name.

        Returns:
            Attribute value from app_state.

        Raises:
            AttributeError: If attribute not found in app_state either.
        """
        if hasattr(self.app_state, name):
            return getattr(self.app_state, name)
        raise AttributeError(f"App object has no attribute {name}")

    def generate_ui(self):
        """
        Generate the Gradio user interface.

        Creates interactive UI components for configuration selection,
        parameter editing, and command generation.

        Returns:
            gr.Blocks: Gradio application instance.
        """
        with gr.Blocks() as app:
            gr.Markdown("# FusionBench Command Generator")

            # 1. Choose a root config file
            with gr.Row(equal_height=True):
                root_configs = gr.Dropdown(
                    choices=self.group_tree.configs,
                    value=self.config_name,
                    label="Root Config",
                    scale=4,
                )
                reset_button = gr.Button("Reset", scale=1)

            with gr.Row():
                with gr.Column(scale=2):
                    command_output = gr.Code(
                        value=self.app_state.generate_command(),
                        language="shell",
                        label="Generated Command",
                        interactive=False,
                    )

                    @gr.render(inputs=[root_configs, command_output])
                    def render_config_groups(config_name, *args):
                        # Generate interactive elements for each group in the config object
                        if not config_name:
                            return gr.Markdown("Select a root config to start.")

                        config = self.app_state.update_config(config_name).config
                        group_tree = self.group_tree

                        def render_group(
                            name: str,
                            config: DictConfig,
                            group_tree: ConfigGroupNode,
                            prefix: str = "",
                        ):
                            gr.Markdown(f"### {prefix}{name}")
                            group_config = gr.Dropdown(
                                choices=group_tree.configs,
                                value=self.get_override(
                                    group_tree.parent.prefix + name
                                ),
                                label="Config File",
                            )
                            group_config.select(
                                lambda c: self.app_state.update_override(
                                    group_tree.parent.prefix + name, c
                                ).config_str_and_command,
                                inputs=[group_config],
                                outputs=[config_output, command_output],
                            )
                            with gr.Row():
                                with gr.Column():
                                    for key in config:
                                        if (
                                            isinstance(
                                                config[key], (str, float, int, bool)
                                            )
                                            or config[key] is None
                                        ):
                                            input_box = gr.Textbox(
                                                value=str(config[key]),
                                                info=str(type(config[key])),
                                                label=group_tree.prefix + key,
                                            )
                                            input_box.submit(
                                                lambda v, k=group_tree.prefix + key: self.app_state.update_override(
                                                    k, v
                                                ).config_str_and_command,
                                                inputs=[input_box],
                                                outputs=[config_output, command_output],
                                            )
                                        else:
                                            gr.Code(
                                                language="yaml",
                                                value=OmegaConf.to_yaml(config[key]),
                                                label=group_tree.prefix + key,
                                            )

                                gr.Code(
                                    OmegaConf.to_yaml(config),
                                    language="yaml",
                                    label=group_tree.prefix,
                                )

                        group_keys = []
                        option_keys = []
                        for key in priority_iterable(
                            config, ["method", "modelpool", "taskpool"]
                        ):
                            if group_tree.has_child_group(key):
                                group_keys.append(key)
                            else:
                                option_keys.append(key)

                        for key in group_keys:
                            render_group(key, config[key], group_tree[key])

                        with gr.Row():
                            with gr.Column():
                                for key in option_keys:
                                    if (
                                        isinstance(config[key], (str, float, int, bool))
                                        or config[key] is None
                                    ):
                                        input_box = gr.Textbox(
                                            value=str(config[key]),
                                            info=str(type(config[key])),
                                            label=group_tree.prefix + key,
                                        )
                                        input_box.submit(
                                            lambda v, k=key: self.app_state.update_override(
                                                k, v
                                            ).config_str_and_command,
                                            inputs=[input_box],
                                            outputs=[config_output, command_output],
                                        )
                                    else:
                                        gr.Code(
                                            language="yaml",
                                            value=OmegaConf.to_yaml(config[key]),
                                            label=group_tree.prefix + key,
                                        )
                            gr.Code(
                                OmegaConf.to_yaml(
                                    {key: config[key] for key in option_keys}
                                ),
                                language="yaml",
                                label="Other Global Options",
                            )

                config_output = gr.Code(
                    value=(
                        OmegaConf.to_yaml(self.config)
                        if self.config is not None
                        else ""
                    ),
                    language="yaml",
                    label="Overall Configuration",
                )

            root_configs.change(
                lambda config_name: self.app_state.update_config(
                    config_name
                ).config_str_and_command,
                inputs=[root_configs],
                outputs=[config_output, command_output],
            )

            def reset_app(config_name):
                # Reset overrides and update config
                self.app_state.overrides = []
                return self.app_state.update_config(config_name).config_str_and_command

            reset_button.click(
                reset_app,
                inputs=[root_configs],
                outputs=[config_output, command_output],
            )
        return app


def parse_args():
    """
    Parse command line arguments for the WebUI.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FusionBench Command Generator")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config directory",
    )
    parser.add_argument(
        "--print-tree",
        action="store_true",
        help="Print the config tree",
    )
    parser.add_argument(
        "--bind-ip",
        type=str,
        default="127.0.0.1",
        help="IP to bind the web UI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web UI",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the web UI",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main entry point for the FusionBench WebUI application.

    Parses arguments, initializes the app, and launches the Gradio interface.
    """
    args = parse_args()

    app = App(args).generate_ui()
    app.launch(
        share=args.share,
        server_name=args.bind_ip,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
