#!/usr/bin/env python3
"""
Script to generate ResNet modelpool configs using Jinja templates.

This script generates YAML configuration files for different ResNet models
and datasets using a Jinja2 template.
"""

import os
from pathlib import Path

from jinja2 import Template

# Template for the ResNet modelpool config
CONFIG_TEMPLATE = """defaults:
  - /dataset/image_classification/train@train_datasets:
      - {{ dataset_name }}
  - /dataset/image_classification/test@val_datasets:
      - {{ dataset_name }}
  - _self_
_target_: fusion_bench.modelpool.ResNetForImageClassificationPool
_recursive_: False
type: transformers
models:
  _pretrained_:
    config_path: {{ config_path }}
    pretrained: true
    dataset_name: {{ dataset_name }}
"""


def generate_config(model_path, dataset_name, output_dir=None):
    """
    Generate a single config file using the template.

    Args:
        config_path (str): The HuggingFace model path (e.g., "microsoft/resnet-18")
        dataset_name (str): The dataset name (e.g., "cifar10")
        output_dir (str, optional): Output directory. If None, uses current directory.

    Returns:
        str: The generated config content
    """
    template = Template(CONFIG_TEMPLATE)
    config_content = template.render(config_path=model_path, dataset_name=dataset_name)

    if output_dir:
        # Extract model name from config_path for filename
        model_name = model_path.split("/")[
            -1
        ]  # e.g., "resnet-18" from "microsoft/resnet-18"
        filename = f"{model_name.replace('-','')}_{dataset_name}.yaml"
        output_path = Path(output_dir) / filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the config file
        with open(output_path, "w") as f:
            f.write(config_content)

        print(f"Generated config: {output_path}")

    return config_content


def generate_all_configs(output_dir=None):
    """
    Generate configs for common ResNet models and datasets.

    Args:
        output_dir (str, optional): Output directory. If None, uses current directory.
    """
    from fusion_bench.constants.clip_vision import TASK_NAMES_TALL20

    if output_dir is None:
        output_dir = Path(__file__).parent

    # Common ResNet models from HuggingFace
    models = [
        "microsoft/resnet-18",
        "microsoft/resnet-50",
        "microsoft/resnet-152",
    ]

    # Common datasets
    datasets = TASK_NAMES_TALL20

    print(f"Generating configs in directory: {output_dir}")

    for model_path in models:
        for dataset in datasets:
            generate_config(model_path, dataset, output_dir)


def main():
    """Main function to run the config generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate ResNet modelpool configs")
    parser.add_argument(
        "--config-path",
        type=str,
        help="HuggingFace model path (e.g., microsoft/resnet-18)",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., cifar10)")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate configs for all common ResNet models and datasets",
    )

    args = parser.parse_args()

    if args.generate_all:
        generate_all_configs(args.output_dir)
    elif args.config_path and args.dataset:
        config_content = generate_config(
            args.config_path, args.dataset, args.output_dir
        )
        if not args.output_dir:
            print("Generated config:")
            print(config_content)
    else:
        parser.print_help()
        print("\nExample usage:")
        print(
            "  python _generate_config.py --config-path microsoft/resnet-18 --dataset cifar10"
        )
        print("  python _generate_config.py --generate-all")
        print("  python _generate_config.py --generate-all --output-dir .")


if __name__ == "__main__":
    main()
