# Select Logger for Experiment Tracking

This tutorial demonstrates how to select and configure different loggers in FusionBench for experiment tracking. By default, FusionBench uses [TensorBoard](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) for logging, which is great for local development and quick experiments, but you can easily switch to other logging backends such as CSV, Weights & Biases (wandb), MLFlow, or SwanLab.

## 🎯 Overview

FusionBench integrates with Lightning Fabric's logging system, which provides a unified interface for various experiment tracking tools. This allows you to:

- Track metrics and hyperparameters across different experiments
- Compare results between different fusion algorithms
- Monitor training progress in real-time
- Store experiment artifacts and configurations

## 📋 Available Loggers

FusionBench supports the following loggers out of the box, the configuration files for which are located in `config/fabric/loggers/`:

| Logger               | Description                            | Configuration File        | Use Case                                   |
| -------------------- | -------------------------------------- | ------------------------- | ------------------------------------------ |
| **TensorBoard**      | Default logger, built-in visualization | `tensorboard_logger.yaml` | Local development, quick experiments       |
| **Weights & Biases** | Cloud-based experiment tracking        | `wandb_logger.yaml`       | Team collaboration, advanced visualization |
| **MLFlow**           | Open-source ML lifecycle platform      | `mlflow_logger.yaml`      | Model registry, experiment comparison      |
| **SwanLab**          | Experiment tracking and visualization  | `swandb_logger.yaml`      | Alternative to wandb                       |
| **CSV**              | Lightweight file-based logging         | `csv_logger.yaml`         | Simple tracking, automated pipelines       |

## ⚙️ Configuration Methods

There are two ways to select a logger in FusionBench:

### Command Line Override (Recommended)

The simplest way is to override the logger configuration via the command line:

```bash
# Use CSV logger
fusion_bench \
    --config-name fabric_model_fusion \
    fabric/loggers=csv_logger \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    method=simple_average

# Use Weights & Biases logger
fusion_bench \
    --config-name fabric_model_fusion \
    fabric/loggers=wandb_logger \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    method=simple_average

# Use SwanLab logger
fusion_bench \
    --config-name fabric_model_fusion \
    fabric/loggers=swandb_logger \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    method=simple_average
```

## 🔧 Logger-Specific Configuration

Each logger has its own configuration file located in `config/fabric/loggers/`. Here's how to customize them:

### TensorBoard Logger

```yaml title="config/fabric/loggers/tensorboard_logger.yaml"
--8<-- "config/fabric/loggers/tensorboard_logger.yaml"
```

**Customization example:**

```bash
fusion_bench \
    --config-name fabric_model_fusion \
    fabric/loggers=tensorboard_logger \
    fabric.loggers.name=my_experiment \
    fabric.loggers.version=v1 \
    method=simple_average
```

### CSV Logger

```yaml title="config/fabric/loggers/csv_logger.yaml"
--8<-- "config/fabric/loggers/csv_logger.yaml"
```

**Customization example:**

```bash
fusion_bench \
    --config-name fabric_model_fusion \
    fabric/loggers=csv_logger \
    fabric.loggers.flush_logs_every_n_steps=50 \
    method=task_arithmetic
```

### Weights & Biases Logger

```yaml title="config/fabric/loggers/wandb_logger.yaml"
--8<-- "config/fabric/loggers/wandb_logger.yaml"
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
    fabric/loggers=wandb_logger \
    fabric.loggers.project=my_fusion_project \
    fabric.loggers.name=experiment_001 \
    method=ties_merging
```

### SwanLab Logger

```yaml title="config/fabric/loggers/swandb_logger.yaml"
--8<-- "config/fabric/loggers/swandb_logger.yaml"
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

## 🎓 Next Steps

- Learn how to [customize algorithms](customize_algorithm.md) with logging capabilities
- Explore [advanced program customization](../advanced_examples/customize_program.md)
- Check out other [intermediate examples](index.md)
