import argparse
import functools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import dearpygui.dearpygui as dpg
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from fusion_bench.scripts.cli import _get_default_config_path

# Keeping the ConfigGroupNode and AppState classes as they are
from fusion_bench.scripts.webui import AppState, ConfigGroupNode, priority_iterable


class App:
    def __init__(self, args):
        self.args = args
        self.group_tree = ConfigGroupNode(self.config_path)
        if args.print_tree:
            print(self.group_tree)

        if "example_config" in self.group_tree.configs:
            self.initial_config_name = "example_config"
        else:
            self.initial_config_name = self.group_tree.configs[0]

        initialize_config_dir(
            config_dir=self.config_path,
            job_name=Path(__file__).stem,
            version_base=None,
        )

        self.app_state = AppState(self.config_path, self.initial_config_name)

    @functools.cached_property
    def config_path(self):
        if self.args.config_path:
            return Path(self.args.config_path)
        else:
            return _get_default_config_path()

    def generate_ui(self):
        dpg.create_context()

        with dpg.window(label="FusionBench Command Generator"):
            dpg.add_text("FusionBench Command Generator")

            def update_config(sender, app_data, user_data):
                self.app_state.update_config(app_data)
                dpg.set_value("config_output", self.app_state.config_str)
                dpg.set_value("command_output", self.app_state.generate_command())
                self.render_config_groups()

            dpg.add_combo(
                label="Root Config",
                items=self.group_tree.configs,
                default_value=self.initial_config_name,
                callback=update_config,
                tag="root_config",
            )

            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_input_text(
                        label="Generated Command",
                        multiline=True,
                        readonly=True,
                        tag="command_output",
                        width=400,
                        height=100,
                    )

                with dpg.group():
                    dpg.add_input_text(
                        label="Overall Configuration",
                        multiline=True,
                        readonly=True,
                        tag="config_output",
                        width=400,
                        height=300,
                    )

            dpg.add_separator()

            with dpg.group(tag="config_groups"):
                self.render_config_groups()

        dpg.create_viewport(
            title="FusionBench Command Generator", width=1000, height=800
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def render_config_groups(self):
        dpg.delete_item("config_groups", children_only=True)

        config = self.app_state.config
        group_tree = self.group_tree

        def render_group(
            name: str, config: DictConfig, group_tree: ConfigGroupNode, prefix: str = ""
        ):
            with dpg.group(parent="config_groups"):
                dpg.add_text(f"### {prefix}{name}")

                def update_group_config(sender, app_data, user_data):
                    self.app_state.update_override(user_data, app_data)
                    dpg.set_value("config_output", self.app_state.config_str)
                    dpg.set_value("command_output", self.app_state.generate_command())

                dpg.add_combo(
                    label="Config File",
                    items=group_tree.configs,
                    default_value=self.app_state.get_override(
                        group_tree.parent.prefix + name
                    ),
                    callback=update_group_config,
                    user_data=group_tree.parent.prefix + name,
                )

                for key in config:
                    if (
                        isinstance(config[key], (str, float, int, bool))
                        or config[key] is None
                    ):

                        def update_value(sender, app_data, user_data):
                            self.app_state.update_override(user_data, app_data)
                            dpg.set_value("config_output", self.app_state.config_str)
                            dpg.set_value(
                                "command_output", self.app_state.generate_command()
                            )

                        dpg.add_input_text(
                            label=group_tree.prefix + key,
                            default_value=str(config[key]),
                            callback=update_value,
                            user_data=group_tree.prefix + key,
                        )
                    else:
                        dpg.add_input_text(
                            label=group_tree.prefix + key,
                            default_value=OmegaConf.to_yaml(config[key]),
                            multiline=True,
                            readonly=True,
                        )

        group_keys = []
        option_keys = []
        for key in priority_iterable(config, ["method", "modelpool", "taskpool"]):
            if group_tree.has_child_group(key):
                group_keys.append(key)
            else:
                option_keys.append(key)

        for key in group_keys:
            render_group(key, config[key], group_tree[key])

        with dpg.group(parent="config_groups"):
            dpg.add_text("Other Global Options")
            for key in option_keys:
                if (
                    isinstance(config[key], (str, float, int, bool))
                    or config[key] is None
                ):

                    def update_value(sender, app_data, user_data):
                        self.app_state.update_override(user_data, app_data)
                        dpg.set_value("config_output", self.app_state.config_str)
                        dpg.set_value(
                            "command_output", self.app_state.generate_command()
                        )

                    dpg.add_input_text(
                        label=group_tree.prefix + key,
                        default_value=str(config[key]),
                        callback=update_value,
                        user_data=group_tree.prefix + key,
                    )
                else:
                    dpg.add_input_text(
                        label=group_tree.prefix + key,
                        default_value=OmegaConf.to_yaml(config[key]),
                        multiline=True,
                        readonly=True,
                    )


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
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    app = App(args)
    app.generate_ui()


if __name__ == "__main__":
    main()
