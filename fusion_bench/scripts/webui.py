import argparse
import functools
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterable

import gradio as gr
import hydra
import yaml
from colorama import Fore, Style  # For cross-platform color support
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from fusion_bench.scripts.cli import _get_default_config_path


class ConfigGroupNode:
    name: str
    path: Path
    parent: Optional["ConfigGroupNode"] = None
    children: List["ConfigGroupNode"]
    configs: List[str]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        assert self.path.is_dir()
        self.name = self.path.stem
        self.children = []
        self.configs = []
        for child in self.path.iterdir():
            if child.is_dir():
                child_node = ConfigGroupNode(child)
                child_node.parent = self
                self.children.append(child_node)
            elif child.is_file() and child.suffix == ".yaml":
                self.configs.append(child.stem)

    def __repr__(self):
        """
        Return string of the tree structure
        """
        return f"{Fore.BLUE}{self.name}{Style.RESET_ALL}\n" + self._repr_indented()

    def _repr_indented(self, prefix=""):
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
        return any(child.name == name for child in self.children)

    def __getitem__(self, key: str) -> Union["ConfigGroupNode", str]:
        for child in self.children:
            if child.name == key:
                return child
        for config in self.configs:
            if config == key:
                return config
        raise KeyError(f"No child group or config named {key}")

    @functools.cached_property
    def prefix(self) -> str:
        if self.parent is None:
            return ""
        return self.parent.prefix + self.name + "."


def priority_iterable(iter, priority_keys):
    items = list(iter)
    for key in priority_keys:
        if key in items:
            yield key
            items.remove(key)
    for item in items:
        yield item


class App:
    def __init__(self, args):
        super().__init__()
        self.args = args
        group_tree = ConfigGroupNode(self.config_path)
        if args.print_tree:
            print(group_tree)

        self.group_tree = group_tree
        self.overrides = []
        self.config = None
        if "example_config" in group_tree.configs:
            self.config_name = "example_config"
        else:
            self.config_name = group_tree.configs[0]

        initialize_config_dir(
            config_dir=self.config_path,
            job_name=Path(__file__).stem,
            version_base=None,
        )

        self.config = self.generate_config(self.config_name)

    @functools.cached_property
    def config_path(self):
        if self.args.config_path:
            return Path(self.args.config_path)
        else:
            return _get_default_config_path()

    def generate_config(self, config_name: str):
        config = compose(config_name=config_name, overrides=self.overrides)
        return config

    def update_config(self, config_name):
        # Generate a DictConfig object from the chosen config file
        self.config_name = config_name
        self.config = config = self.generate_config(config_name)

        # Generate the command according to `config_name` and `overrides` (a list of strings)
        command = "fusion_bench \\\n"
        if self.args.config_path is not None:
            command += f"--config-path {self.config_path} \\\n"
        command += f"--config-name {config_name} \\\n"
        command += " \\\n".join(self.overrides)
        command = command.strip()
        command = command.strip("\\")
        command = command.strip()

        return OmegaConf.to_yaml(config), command

    def get_override(self, key):
        for ov in self.overrides:
            if ov.startswith(f"{key}="):
                return "".join(ov.split("=")[1:])
        return None

    def update_override(self, key, value):
        self.overrides = [ov for ov in self.overrides if not ov.startswith(f"{key}=")]
        if value:
            self.overrides.append(f"{key}={value}")
        return self.update_config(self.config_name)

    def generate_ui(self):
        with gr.Blocks() as app:
            gr.Markdown("# FusionBench Command Generator")

            # 1. Choose a root config file
            root_configs = gr.Dropdown(
                choices=self.group_tree.configs,
                value=self.config_name,
                label="Root Config",
            )

            with gr.Row():
                with gr.Column(scale=2):
                    command_output = gr.Code(
                        language="shell", label="Generated Command"
                    )

                    @gr.render(inputs=[root_configs, command_output])
                    def render_config_groups(config_name, *args):
                        # Generate interactive elements for each group in the config object
                        if not config_name:
                            return gr.Markdown("Select a root config to start.")

                        config = self.generate_config(config_name)
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
                                lambda c: self.update_override(
                                    group_tree.parent.prefix + name, c
                                ),
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
                                                lambda v, k=group_tree.prefix + key: self.update_override(
                                                    k, v
                                                ),
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
                                            lambda v, k=key: self.update_override(k, v),
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
                self.update_config,
                inputs=[root_configs],
                outputs=[config_output, command_output],
            )

        return app


def parse_args():
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
    args = parse_args()

    app = App(args).generate_ui()
    app.launch(
        share=args.share,
        server_name=args.bind_ip,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
