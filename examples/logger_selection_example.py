#!/usr/bin/env python3
"""
Example demonstrating different logger configurations in FusionBench.

This script shows how to use different loggers (TensorBoard, CSV, Wandb, etc.)
for experiment tracking in FusionBench.

Usage:
    # Run with TensorBoard logger (default)
    python logger_selection_example.py

    # Run with CSV logger
    python logger_selection_example.py --logger csv_logger

    # Run with Weights & Biases logger (requires wandb installation and login)
    python logger_selection_example.py --logger wandb_logger
"""
import argparse
import os

import lightning as L
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()


def run_with_logger(logger_config: str = "tensorboard_logger"):
    """
    Run model fusion with specified logger configuration.

    Args:
        logger_config: Name of the logger configuration to use.
                      Options: 'tensorboard_logger', 'csv_logger', 'wandb_logger',
                      'mlflow_logger', 'swandb_logger'
    """
    print(f"\n{'='*60}")
    print(f"Running with logger: {logger_config}")
    print(f"{'='*60}\n")

    # Load configuration using Hydra
    with initialize(
        version_base=None,
        config_path=os.path.relpath(
            _get_default_config_path(), start=os.path.dirname(__file__)
        ),
    ):
        cfg = compose(
            config_name="fabric_model_fusion",
            overrides=[
                f"fabric.loggers={logger_config}",
                "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
                "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8",
            ],
        )

        # Instantiate fabric with the selected logger
        fabric = instantiate(cfg.fabric)
        if not fabric.is_launched:
            fabric.launch()

        print(f"Fabric logger type: {type(fabric.logger).__name__}")
        print(f"Log directory: {fabric.logger.log_dir if hasattr(fabric.logger, 'log_dir') else 'N/A'}")

        # Load models and tasks
        modelpool = instantiate(cfg.modelpool)
        taskpool = instantiate(cfg.taskpool, move_to_device=False)
        taskpool.fabric = fabric

        # Run fusion algorithm
        print("\nRunning Simple Average algorithm...")
        algorithm = SimpleAverageAlgorithm()
        merged_model = algorithm.run(modelpool)

        # Log some custom metrics to demonstrate logger usage
        print("\nLogging custom metrics...")
        fabric.log("custom/model_count", len(modelpool.model_names))
        fabric.log_dict({
            "custom/fusion_method": 0,  # 0 for simple average
            "custom/total_parameters": sum(p.numel() for p in merged_model.parameters()),
        })

        # Evaluate
        print("\nEvaluating merged model...")
        report = taskpool.evaluate(merged_model)

        print(f"\nResults with {logger_config}:")
        print(report)

        # Show where logs are saved
        if hasattr(fabric.logger, 'log_dir'):
            print(f"\n✓ Logs saved to: {fabric.logger.log_dir}")
        elif hasattr(fabric.logger, 'save_dir'):
            print(f"\n✓ Logs saved to: {fabric.logger.save_dir}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate logger selection in FusionBench"
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard_logger",
        choices=["tensorboard_logger", "csv_logger", "wandb_logger", "mlflow_logger", "swandb_logger"],
        help="Logger configuration to use (default: tensorboard_logger)",
    )
    args = parser.parse_args()

    try:
        run_with_logger(args.logger)
        print("\n✓ Example completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Some loggers require additional setup:")
        print("  - wandb_logger: pip install wandb && wandb login")
        print("  - mlflow_logger: pip install mlflow")
        print("  - swandb_logger: pip install swanlab")
        raise


if __name__ == "__main__":
    main()
