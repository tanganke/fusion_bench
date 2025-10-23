# Select Logger for Experiment Tracking

This tutorial demonstrates how to select and configure different loggers in FusionBench for experiment tracking. By default, FusionBench uses TensorBoard for logging, but you can easily switch to other logging backends such as CSV, Weights & Biases (wandb), MLFlow, or SwanLab.

## 🎯 Overview

FusionBench integrates with [Lightning Fabric's logging system](https://lightning.ai/docs/fabric/stable/guide/loggers.html), which provides a unified interface for various experiment tracking tools. This allows you to:

- Track metrics and hyperparameters across different experiments
- Compare results between different fusion algorithms
- Monitor training progress in real-time
- Store experiment artifacts and configurations

## 📋 Available Loggers

FusionBench supports the following loggers out of the box:

| Logger | Description | Configuration File | Use Case |
|--------|-------------|-------------------|----------|
| **TensorBoard** | Default logger, built-in visualization | `tensorboard_logger.yaml` | Local development, quick experiments |
| **CSV** | Lightweight file-based logging | `csv_logger.yaml` | Simple tracking, automated pipelines |
| **Weights & Biases** | Cloud-based experiment tracking | `wandb_logger.yaml` | Team collaboration, advanced visualization |
| **MLFlow** | Open-source ML lifecycle platform | `mlflow_logger.yaml` | Model registry, experiment comparison |
| **SwanLab** | Experiment tracking and visualization | `swandb_logger.yaml` | Alternative to wandb |

## ⚙️ Configuration Methods

There are two ways to select a logger in FusionBench:

### Method 1: Command Line Override (Recommended)

The simplest way is to override the logger configuration via the command line:

```bash
# Use CSV logger
fusion_bench \
    --config-name fabric_model_fusion \
    fabric.loggers=csv_logger \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    method=simple_average

# Use Weights & Biases logger
fusion_bench \
    --config-name fabric_model_fusion \
    fabric.loggers=wandb_logger \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    method=simple_average

# Use SwanLab logger
fusion_bench \
    --config-name fabric_model_fusion \
    fabric.loggers=swandb_logger \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    method=simple_average
```

### Method 2: Modify Configuration File

You can also modify the default configuration in your config file. Edit the fabric configuration's defaults section:

```yaml title="config/fabric_model_fusion.yaml"
defaults:
  - loggers: csv_logger  # Change this line to select a different logger
  - _self_
```

Or create your own custom configuration file:

```yaml title="config/my_custom_fabric.yaml"
defaults:
  - loggers: wandb_logger  # Your preferred logger
  - _self_

_target_: lightning.Fabric
_recursive_: true
devices: 1
strategy: auto
accelerator: auto
```

## 🔧 Logger-Specific Configuration

Each logger has its own configuration file located in `fusion_bench_config/fabric/loggers/`. Here's how to customize them:

### TensorBoard Logger

```yaml title="fusion_bench_config/fabric/loggers/tensorboard_logger.yaml"
_target_: lightning.fabric.loggers.TensorBoardLogger
root_dir: ${path.log_dir}  # Root directory for logs
name: ""                    # Experiment name
version: ""                 # Experiment version
sub_dir: null               # Subdirectory within root_dir
default_hp_metric: false    # Whether to log default hyperparameter metric
```

**Customization example:**

```bash
fusion_bench \
    --config-name fabric_model_fusion \
    fabric.loggers=tensorboard_logger \
    fabric.loggers.name=my_experiment \
    fabric.loggers.version=v1 \
    method=simple_average
```

### CSV Logger

```yaml title="fusion_bench_config/fabric/loggers/csv_logger.yaml"
_target_: lightning.fabric.loggers.CSVLogger
root_dir: ${path.log_dir}
name: ""
version: ""
prefix: ""                        # Prefix for CSV filenames
flush_logs_every_n_steps: 100     # How often to write to disk
```

**Customization example:**

```bash
fusion_bench \
    --config-name fabric_model_fusion \
    fabric.loggers=csv_logger \
    fabric.loggers.flush_logs_every_n_steps=50 \
    method=task_arithmetic
```

### Weights & Biases Logger

```yaml title="fusion_bench_config/fabric/loggers/wandb_logger.yaml"
_target_: wandb.integration.lightning.fabric.WandbLogger
project: ${hydra:job.config_name}  # Project name in wandb
save_dir: ${path.log_dir}
```

**Prerequisites:** Install wandb and login:

```bash
pip install wandb
wandb login
```

**Customization example:**

```bash
fusion_bench \
    --config-name fabric_model_fusion \
    fabric.loggers=wandb_logger \
    fabric.loggers.project=my_fusion_project \
    fabric.loggers.name=experiment_001 \
    method=ties_merging
```

### SwanLab Logger

```yaml title="fusion_bench_config/fabric/loggers/swandb_logger.yaml"
_target_: swandb.integration.pytorch_lightning.SwanLabLogger
project: ${hydra:job.config_name}
description: "SwanLab logger with FusionBench"
save_dir: ${path.log_dir}
```

**Prerequisites:** Install swanlab:

```bash
pip install swanlab
```

## 🐍 Programmatic Usage

You can also configure loggers programmatically when using FusionBench as a library:

```python
import os
import lightning as L
from lightning.fabric.loggers import CSVLogger
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()

# Option 1: Create a logger directly
csv_logger = CSVLogger(
    root_dir="outputs/logs",
    name="my_experiment",
    flush_logs_every_n_steps=50
)

fabric = L.Fabric(
    accelerator="auto",
    devices=1,
    loggers=csv_logger
)
fabric.launch()

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
            "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
            "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8",
        ],
    )
    modelpool: CLIPVisionModelPool = instantiate(cfg.modelpool)
    taskpool: CLIPVisionModelTaskPool = instantiate(cfg.taskpool, move_to_device=False)
    taskpool.fabric = fabric

# Run the fusion algorithm
algorithm = SimpleAverageAlgorithm()
merged_model = algorithm.run(modelpool)

# Evaluate the merged model
report = taskpool.evaluate(merged_model)
print(report)
```

For Weights & Biases:

```python
from lightning.fabric.loggers import WandbLogger

# Create wandb logger with custom configuration
wandb_logger = WandbLogger(
    project="my_fusion_project",
    name="experiment_001",
    save_dir="outputs/logs",
    log_model=True  # Log model checkpoints
)

fabric = L.Fabric(
    accelerator="auto",
    devices=1,
    loggers=wandb_logger
)
fabric.launch()

# Continue with your fusion workflow...
```

## 📊 Accessing Logger in Your Algorithm

If you're implementing a custom algorithm using `LightningFabricMixin`, you can access the logger to log custom metrics:

```python
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin

class MyCustomAlgorithm(BaseAlgorithm, LightningFabricMixin):
    
    def run(self, modelpool):
        # Log scalar metrics
        self.fabric.log("my_metric", 0.95)
        
        # Log multiple metrics at once
        self.fabric.log_dict({
            "accuracy": 0.95,
            "loss": 0.05,
            "f1_score": 0.93
        })
        
        # Log hyperparameters
        self.log_hyperparams()
        
        # Access the underlying logger for advanced features
        logger = self.fabric.logger
        
        # For TensorBoard logger specifically
        if hasattr(self, 'tensorboard_summarywriter'):
            writer = self.tensorboard_summarywriter
            # Log custom data (images, histograms, etc.)
            # writer.add_image(...)
            # writer.add_histogram(...)
        
        # Your fusion logic here...
        merged_model = ...
        
        return merged_model
```

## 🚀 Complete Example

Here's a complete example script that demonstrates logger selection:

```python title="examples/logger_selection_example.py"
#!/usr/bin/env python3
"""
Example demonstrating different logger configurations in FusionBench.
"""
import os
import lightning as L
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()

def run_with_logger(logger_config: str):
    """Run fusion with specified logger configuration."""
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
        
        # Load models and tasks
        modelpool = instantiate(cfg.modelpool)
        taskpool = instantiate(cfg.taskpool, move_to_device=False)
        taskpool.fabric = fabric
        
        # Run fusion algorithm
        algorithm = SimpleAverageAlgorithm()
        merged_model = algorithm.run(modelpool)
        
        # Evaluate
        report = taskpool.evaluate(merged_model)
        print(f"\nResults with {logger_config}:")
        print(report)
        
        return report

if __name__ == "__main__":
    # Try different loggers
    loggers = ["tensorboard_logger", "csv_logger"]
    
    # Uncomment if you have wandb configured
    # loggers.append("wandb_logger")
    
    for logger_config in loggers:
        try:
            run_with_logger(logger_config)
        except Exception as e:
            print(f"Error with {logger_config}: {e}")
```

## 📝 Tips and Best Practices

1. **Local Development**: Use TensorBoard or CSV logger for quick experiments and debugging
2. **Team Collaboration**: Use Weights & Biases or MLFlow for sharing results with your team
3. **Production Pipelines**: CSV logger is lightweight and reliable for automated workflows
4. **Cloud vs Local**: 
   - TensorBoard and CSV loggers store data locally
   - Wandb, MLFlow, and SwanLab can sync to cloud services
5. **Multiple Loggers**: You can use multiple loggers simultaneously by passing a list in the Fabric configuration (advanced usage)

## 🔗 Related Resources

- [Lightning Fabric Logger Documentation](https://lightning.ai/docs/fabric/stable/guide/loggers.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [FusionBench Lightning Fabric Mixin](../../guides/fusion_bench/mixins/lightning_fabric.md)

## 🎓 Next Steps

- Learn how to [customize algorithms](customize_algorithm.md) with logging capabilities
- Explore [advanced program customization](../advanced_examples/customize_program.md)
- Check out other [intermediate examples](index.md)
