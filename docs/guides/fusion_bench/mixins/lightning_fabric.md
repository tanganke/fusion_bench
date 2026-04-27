# LightningFabricMixin

The `LightningFabricMixin` provides deep integration with PyTorch Lightning Fabric, enabling distributed computing, device management, and structured logging for FusionBench algorithms. This mixin abstracts away the complexity of multi-GPU training, data parallelism, and experiment tracking, allowing algorithm implementations to focus on their core logic.

## Overview

Lightning Fabric is a lightweight library from the Lightning ecosystem that provides a clean API for distributed training without the overhead of full framework abstractions like PyTorch Lightning's `LightningModule`. The mixin integrates Fabric into FusionBench's three-component architecture, specifically into the **Method** (algorithm) layer.

## Key Features

### 1. Distributed Computing

The mixin handles distributed setup automatically. When you call `self.fabric` from your algorithm, it initializes a `Lightning Fabric` instance based on the YAML configuration.

```python
class MyAlgorithm(LightningFabricMixin, BaseAlgorithm):
    def run(self, modelpool):
        # Models are automatically placed on the correct device
        model = modelpool.load_model("model_a")
        model = self.to_device(model)

        # Fabric handles gradient synchronization in distributed settings
        with self.fabric.init_module():
            model = self.fabric.setup(model)
```

### 2. Device Management

The `to_device()` method moves tensors or modules to the appropriate device (GPU, TPU, or CPU) as configured by Fabric:

```python
# Move a model to the correct device
model = self.to_device(model)

# Move a tensor to the correct device
tensor = self.to_device(torch.randn(32, 768))
```

### 3. Logging

The mixin provides multiple logging methods:

```python
# Log a single metric
self.log("train/loss", loss_value, step=epoch)

# Log multiple metrics at once
self.log_dict({"train/acc": acc, "train/loss": loss}, step=epoch)

# Log optimizer learning rates
self.log_optimizer_lr(optimizer, step=epoch)

# Log hyperparameters (saves to YAML)
self.log_hyperparams(self.config)
```

### 4. Hyperparameter Persistence

The `log_hyperparams()` method saves the full Hydra configuration as a YAML file in the log directory. This is decorated with `@rank_zero_only`, so it only executes on the main process in distributed settings.

```python
# In your algorithm's on_run_start() hook:
def on_run_start(self):
    super().on_run_start()
    self.log_hyperparams()  # Saves config.yaml to log directory
```

### 5. Log Directory Management

The `log_dir` property retrieves the log directory from the configured logger. It handles multiple logger types:

- **TensorBoardLogger**: Returns `logger.log_dir`
- **SwanLabLogger**: Returns `logger.save_dir` or `logger._logdir`
- **MLFlowLogger**: Returns `logger.save_dir` or falls back to program config

## Usage Patterns

### Basic Usage

The simplest way to use the mixin is through multiple inheritance:

```python
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.method import BaseAlgorithm

class MyAlgorithm(LightningFabricMixin, BaseAlgorithm):
    def run(self, modelpool):
        model = modelpool.load_model("_pretrained_")
        model = self.to_device(model)
        model = self.fabric.setup(model)
        # ... your algorithm logic ...
        return model
```

### With Configuration

The mixin reads the `fabric` section from your Hydra config:

```yaml
# config/method/my_algorithm.yaml
_target_: fusion_bench.method.MyAlgorithm
fabric:
  _target_: lightning.Fabric
  accelerator: gpu
  devices: 2
  strategy: ddp
  precision: "16-mixed"
```

### With TensorBoard

To enable TensorBoard logging:

```yaml
fabric:
  _target_: lightning.Fabric
  accelerator: gpu
  devices: 1
  loggers:
    - _target_: lightning.fabric.loggers.TensorBoardLogger
      root_dir: outputs
      name: tensorboard
```

Then access the SummaryWriter:

```python
writer = self.tensorboard_summarywriter
writer.add_scalar("train/loss", loss, step=epoch)
```

## Advanced Features

### FSDP Support

For large models that require Fully Sharded Data Parallel training:

```python
from fusion_bench.mixins.lightning_fabric import get_policy, get_size_based_auto_wrap_policy

policy = get_size_based_auto_wrap_policy(min_num_params=100_000_000)
```

### MLFlow Artifact Logging

When using MLFlow as the logger, you can log files and directories as artifacts:

```python
# Log a single file
self.log_artifact(local_path="model.pth", artifact_path="checkpoints")

# Log a directory of files
self.log_artifacts(local_dir="outputs/plots", artifact_path="figures")
```

### Debug Mode

Check if the program is running in debug (fast_dev_run) mode:

```python
if self.is_debug_mode:
    # Run with reduced iterations for debugging
    max_epochs = 1
```

## Lifecycle Hooks

The mixin provides a `finalize()` method that should be called during cleanup:

```python
def on_run_end(self):
    super().on_run_end()
    self.finalize()  # Properly clean up Fabric resources
```

The `__del__` destructor also calls `finalize()` automatically, ensuring proper cleanup even if not explicitly called. For MLFlow loggers, `finalize()` sets the run status to "success" or "failed" based on whether an exception occurred.

## Important Notes

1. **Mixin order matters**: Always place `LightningFabricMixin` before `BaseAlgorithm` in the inheritance list.
2. **Lazy initialization**: The Fabric instance is created lazily on first access via `self.fabric`. If no `fabric` config is found, a default single-device Fabric is created.
3. **CLI detection**: If running via the Lightning CLI (`lightning run`), the mixin skips calling `launch()` since the CLI handles that.
4. **Rank zero operations**: Methods like `log_hyperparams` are decorated with `@rank_zero_only`, so they only execute on the main process.
5. **Config integration**: The log directory from the logger is automatically injected into `config.log_dir` if not already set.

## Reference

::: fusion_bench.mixins.lightning_fabric.LightningFabricMixin
::: fusion_bench.mixins.lightning_fabric._fabric_has_logger
::: fusion_bench.mixins.lightning_fabric.get_policy
::: fusion_bench.mixins.lightning_fabric.get_size_based_auto_wrap_policy
