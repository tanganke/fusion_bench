# Creating a Custom TaskPool

A TaskPool is responsible for evaluating fused models on specific tasks or datasets. It defines the evaluation logic, metrics, and data loading pipeline. This guide shows you how to build a custom TaskPool from scratch.

## Understanding the BaseTaskPool

The base class `fusion_bench.taskpool.base_pool.BaseTaskPool` provides the foundation. Its key characteristics:

- **Abstract `evaluate` method**: Subclasses must implement `evaluate(self, model, *args, **kwargs) -> Dict[str, Any]`.
- **YAML serialization**: Inherits from `BaseYAMLSerializable` for config management.
- **Return format**: The `evaluate` method returns a dictionary mapping task names to metric dictionaries.

### The `evaluate` Method Signature

```python
from abc import abstractmethod
from typing import Any, Dict

class BaseTaskPool(BaseYAMLSerializable):
    _program = None
    _config_key = "taskpool"

    @abstractmethod
    def evaluate(self, model: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate the model and return a structured report."""
        pass
```

The expected return format is:

```python
{
    "model_info": {
        "trainable_params": 1234567,
        "all_params": 1234567,
        "trainable_percentage": 1.0,
    },
    "task_name_1": {
        "accuracy": 0.95,
        "loss": 0.15,
    },
    "task_name_2": {
        "accuracy": 0.87,
        "loss": 0.42,
    },
    "average": {
        "accuracy": 0.91,
        "loss": 0.285,
    }
}
```

## Step 1: Inherit from BaseTaskPool

Create a new file in `fusion_bench/taskpool/`:

```python
from typing import Any, Dict

from fusion_bench import BaseTaskPool


class MyCustomTaskPool(BaseTaskPool):
    """A custom task pool for evaluating on my specific tasks."""

    def __init__(self, test_datasets: dict, **kwargs):
        super().__init__(**kwargs)
        self._test_datasets = test_datasets
```

## Step 2: Implement the `evaluate` Method

The `evaluate` method is the core of your TaskPool. It receives the fused model and must return a metrics dictionary.

### Minimal Example

```python
from torch import nn
import torch


class MyCustomTaskPool(BaseTaskPool):
    def __init__(self, test_datasets: dict, **kwargs):
        super().__init__(**kwargs)
        self._test_datasets = test_datasets

    def evaluate(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Evaluate the model on each configured test dataset."""
        report = {}

        for task_name, dataset_config in self._test_datasets.items():
            # Load dataset (you can use the instantiate utility)
            dataset = instantiate(dataset_config)

            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=False
            )

            # Run evaluation
            task_result = self._run_evaluation(model, dataloader)
            report[task_name] = task_result

        # Compute average
        if len(report) > 0:
            report["average"] = self._compute_average(report)

        return report

    def _run_evaluation(self, model, dataloader) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = self._compute_loss(outputs, targets)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item()

        return {
            "accuracy": correct / total,
            "loss": total_loss / len(dataloader),
        }

    def _compute_loss(self, outputs, targets):
        import torch.nn.functional as F
        return F.cross_entropy(outputs, targets)

    @staticmethod
    def _compute_average(report: Dict[str, Any]) -> Dict[str, float]:
        accuracies = [v["accuracy"] for v in report.values() if "accuracy" in v]
        losses = [v["loss"] for v in report.values() if "loss" in v]
        avg = {}
        if accuracies:
            avg["accuracy"] = sum(accuracies) / len(accuracies)
        if losses:
            avg["loss"] = sum(losses) / len(losses)
        return avg
```

### Real Example: ImageClassificationTaskPool

The `ImageClassificationTaskPool` in `fusion_bench/taskpool/image_classification.py` demonstrates a production implementation with Lightning Fabric integration:

```python
from fusion_bench import BaseTaskPool, LightningFabricMixin, auto_register_config

@auto_register_config
class ImageClassificationTaskPool(LightningFabricMixin, BaseTaskPool):

    def __init__(
        self,
        test_datasets: DictConfig | Dict[str, Any],
        *,
        processor: DictConfig | Any,
        dataloader_kwargs: DictConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def evaluate(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        classifier = self.fabric.to_device(model)
        classifier.eval()
        report = {}

        for task_name, test_dataloader in self.test_dataloaders.items():
            result = self._evaluate(classifier, test_dataloader, task_name)
            report[task_name] = result

        report["average"] = self._compute_average(report)
        return report
```

Key patterns used in the real implementation:

- **`LightningFabricMixin`**: Provides device management and distributed training support. Access via `self.fabric`.
- **`@auto_register_config`**: Automatically registers `__init__` parameters into `_config_mapping`.
- **`torch.no_grad()`**: Used in `_evaluate` to disable gradient computation during inference.
- **`tqdm` progress bars**: For tracking evaluation progress.

## Step 3: Add Model Summary Information

Most TaskPools include basic model information in their report:

```python
from fusion_bench.utils.parameters import count_parameters

def evaluate(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
    training_params, all_params = count_parameters(model)

    report = {
        "model_info": {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }
    }

    # ... add task results ...
    return report
```

## Step 4: Configure Test Datasets in YAML

TaskPool configurations live under `config/taskpool/`. The `test_datasets` key defines which datasets to evaluate on.

### Simple TaskPool Config

```yaml
# config/taskpool/my_custom_task.yaml
_target_: fusion_bench.taskpool.MyCustomTaskPool

test_datasets:
  dataset_a:
    _target_: torchvision.datasets.CIFAR10
    root: ./data
    train: false
    download: true
  dataset_b:
    _target_: torchvision.datasets.MNIST
    root: ./data
    train: false
    download: true
```

### Using Hydra Defaults (recommended for complex setups)

For shared dataset configurations, use Hydra defaults:

```yaml
# config/taskpool/CLIPVisionModelTaskPool/clip-vit-classification_TA8_val.yaml
defaults:
  - CLIPVisionModelTaskPool@: _template
  - /dataset/image_classification/val@test_datasets:
      - sun397
      - stanford-cars
      - resisc45
      - eurosat
      - svhn
      - gtsrb
      - mnist
      - dtd
```

The `CLIPVisionModelTaskPool@: _template` line references a template config that defines the TaskPool class and shared settings. The `/dataset/image_classification/val@test_datasets` line injects dataset configs into the `test_datasets` key.

## Complete Working Example

Here is a full TaskPool for text classification:

```python
# fusion_bench/taskpool/text_classification.py
import json
import os
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from fusion_bench import BaseTaskPool, get_rankzero_logger, instantiate
from fusion_bench.utils.parameters import count_parameters

log = get_rankzero_logger(__name__)


class TextClassificationTaskPool(BaseTaskPool):
    """Task pool for text classification evaluation."""

    def __init__(
        self,
        test_datasets: Dict[str, Any],
        dataloader_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._test_datasets = test_datasets
        self.dataloader_kwargs = dataloader_kwargs or {"batch_size": 32, "shuffle": False}
        self._is_setup = False

    def setup(self):
        """Pre-load datasets and dataloaders."""
        self.test_dataloaders = {}
        for name, ds_config in self._test_datasets.items():
            dataset = instantiate(ds_config)
            self.test_dataloaders[name] = DataLoader(dataset, **self.dataloader_kwargs)
        self._is_setup = True

    def evaluate(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        if not self._is_setup:
            self.setup()

        model.eval()
        report = {}

        # Model info
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }

        # Evaluate on each task
        for task_name, dataloader in tqdm(
            self.test_dataloaders.items(), desc="Evaluating tasks"
        ):
            result = self._evaluate_task(model, dataloader)
            report[task_name] = result

        # Average
        accuracies = [v["accuracy"] for v in report.values() if isinstance(v, dict) and "accuracy" in v]
        if accuracies:
            report["average"] = {"accuracy": sum(accuracies) / len(accuracies)}

        log.info(f"Evaluation Result: {report}")
        return report

    def _evaluate_task(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                outputs = model(input_ids=inputs, attention_mask=attention_mask)
                logits = outputs.logits
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {"accuracy": correct / total}
```

With config:

```yaml
# config/taskpool/TextClassificationTaskPool/glue.yaml
_target_: fusion_bench.taskpool.text_classification.TextClassificationTaskPool

test_datasets:
  sst2:
    _target_: datasets.load_dataset
    path: glue
    name: sst2
    split: validation
  mnli:
    _target_: datasets.load_dataset
    path: glue
    name: mnli
    split: validation_matched

dataloader_kwargs:
  batch_size: 32
  shuffle: false
```

## Using LightningFabricMixin for Distributed Evaluation

When your evaluation involves large datasets or GPUs, add `LightningFabricMixin` as the first mixin:

```python
from fusion_bench import BaseTaskPool, LightningFabricMixin, auto_register_config

@auto_register_config
class DistributedEvaluationTaskPool(LightningFabricMixin, BaseTaskPool):
    def __init__(self, test_datasets: dict, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        # Move model to correct device
        model = self.fabric.to_device(model)
        model.eval()

        # Setup dataloaders for distributed inference
        dataloader = self.fabric.setup_dataloaders(self.test_dataloader)

        # ... run evaluation ...
```

Note the mixin order: `LightningFabricMixin` comes before `BaseTaskPool` so its `__init__` runs first.

## Best Practices

1. **Return consistent structure**: Always include `model_info` and `average` keys in your report.
2. **Use `torch.no_grad()`**: Never compute gradients during evaluation.
3. **Use progress bars**: `tqdm` provides visibility into long evaluations.
4. **Log results**: Use `get_rankzero_logger` to log on the main process only.
5. **Separate setup from evaluate**: Pre-load datasets in a `setup()` method to avoid reloading on each call.
6. **Handle device placement**: Use `self.fabric.to_device()` when using `LightningFabricMixin`.

## Troubleshooting

- **`NotImplementedError` on `evaluate`**: Ensure your subclass implements the method (do not forget the `@abstractmethod` contract).
- **Missing test datasets**: Check that your YAML config includes `test_datasets` and each dataset config has a valid `_target_`.
- **Device mismatch**: If using GPUs, ensure both model and tensors are on the same device.

## Next Steps

- See `fusion_bench/taskpool/dummy.py` for the simplest possible TaskPool (model introspection only).
- See `fusion_bench/taskpool/image_classification.py` for a full image classification TaskPool.
- Read the [Custom ModelPool](custom_modelpool.md) guide to create a matching ModelPool.
