# Creating a Custom ModelPool

A ModelPool is the component in FusionBench responsible for managing a collection of models and their associated datasets. It handles model loading, instantiation, and serialization. This guide walks you through creating a custom ModelPool from scratch.

## Understanding the BaseModelPool

The base class `fusion_bench.modelpool.base_pool.BaseModelPool` provides the foundation for all model pools. Key features it provides out of the box:

- **Model registry**: A dictionary (`_models`) mapping model names to configurations or instances.
- **Dataset management**: Optional `train_datasets`, `val_datasets`, and `test_datasets` dictionaries.
- **Special model names**: Support for `_pretrained_` (base/pretrained model) and `_merged_` (merged model) keys.
- **YAML serialization**: Inherits from `BaseYAMLSerializable` for config-to-object conversion.
- **Hydra integration**: Inherits from `HydraConfigMixin` for seamless Hydra configuration handling.

### Key Methods in BaseModelPool

| Method | Purpose |
|--------|---------|
| `load_model(model_name_or_config)` | Load a model by name (from `_models`) or from a direct DictConfig |
| `save_model(model, path)` | Save model state dict to `path` via `torch.save` |
| `add_model(model_name, model_or_config)` | Add a model to the pool at runtime |
| `get_model_config(model_name)` | Get the raw config for a model |
| `models()` | Generator yielding all regular (non-special) models |
| `named_models()` | Generator yielding `(name, model)` tuples |
| `load_pretrained_model()` | Load the model registered under `_pretrained_` |

## Step 1: Inherit from BaseModelPool

Create a new Python file in `fusion_bench/modelpool/`. Your class must inherit from `BaseModelPool`:

```python
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig
from torch import nn

from fusion_bench import BaseModelPool


class MyCustomModelPool(BaseModelPool):
    """A custom model pool for my specific model type."""

    def __init__(
        self,
        models: DictConfig,
        *,
        some_custom_param: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        self.some_custom_param = some_custom_param
```

## Step 2: Implement `load_model`

The `load_model` method is the heart of any ModelPool. It takes a model name (string) or a DictConfig and returns an instantiated `nn.Module`.

The base implementation already handles three cases:

1. **String name in `_models`** with a `dict`/`DictConfig` config -> calls `instantiate()` with `_target_`.
2. **Pre-instantiated `nn.Module`** in `_models` -> returns it directly.
3. **Direct DictConfig** passed as argument -> calls `instantiate()`.

Override this method when you need custom loading logic (e.g., calling `from_pretrained`, resolving platform-specific paths, or applying post-processing):

```python
from typing_extensions import override

@override
def load_model(
    self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
) -> nn.Module:
    """Load a model from the pool with custom logic."""
    if isinstance(model_name_or_config, str) and model_name_or_config in self._models:
        model_name = model_name_or_config
        model_config = self._models[model_name]

        if isinstance(model_config, str):
            # String path - use it as a HuggingFace model ID or local path
            model = MyModelClass.from_pretrained(model_config, *args, **kwargs)
            return model

        if isinstance(model_config, nn.Module):
            # Already instantiated
            return model_config

        # For dict/DictConfig, delegate to parent
        return super().load_model(model_name_or_config, *args, **kwargs)

    return super().load_model(model_name_or_config, *args, **kwargs)
```

### Real Example: CLIPVisionModelPool

The `CLIPVisionModelPool` in `fusion_bench/modelpool/clip_vision/modelpool.py` demonstrates a production-ready override:

```python
@override
def load_model(self, model_name_or_config, *args, **kwargs) -> CLIPVisionModel:
    if isinstance(model_name_or_config, str) and model_name_or_config in self._models:
        match self._models[model_name_or_config]:
            case str() as model_path:
                # Resolve path (supports HuggingFace and ModelScope)
                repo_path = resolve_repo_path(model_path, repo_type="model",
                                              platform=self._platform)
                return CLIPVisionModel.from_pretrained(repo_path, *args, **kwargs)
            case nn.Module() as model:
                return model
            case _:
                return super().load_model(model_name_or_config, *args, **kwargs)
    return super().load_model(model_name_or_config, *args, **kwargs)
```

## Step 3: Optionally Override `save_model`

The default `save_model` uses `torch.save(model.state_dict(), path)`. Override it when your models require a different serialization format:

```python
@override
def save_model(self, model: nn.Module, path: str, *args, **kwargs):
    """Save the model using HuggingFace format."""
    model.save_pretrained(path)
```

## Step 4: Register `_config_mapping` (if adding new attributes)

If your ModelPool has custom attributes that should be serialized to/from YAML, add them to `_config_mapping`:

```python
class MyCustomModelPool(BaseModelPool):
    _config_mapping = BaseModelPool._config_mapping | {
        "_custom_processor": "processor",
        "_platform": "platform",
    }

    def __init__(self, models: DictConfig, *,
                 custom_processor: Optional[DictConfig] = None,
                 platform: str = "hf",
                 **kwargs):
        super().__init__(models, **kwargs)
        self._custom_processor = custom_processor
        self._platform = platform
```

Alternatively, use the `@auto_register_config` decorator to auto-register all `__init__` parameters:

```python
from fusion_bench.mixins import auto_register_config

@auto_register_config
class MyCustomModelPool(BaseModelPool):
    def __init__(self, models, custom_processor: Optional[DictConfig] = None,
                 platform: str = "hf", **kwargs):
        super().__init__(models=models, **kwargs)
        self.custom_processor = custom_processor
        self.platform = platform
```

## Step 5: Create the YAML Configuration

Every ModelPool needs a matching YAML config file. Place it under `config/modelpool/your_pool_name/`:

```yaml
# config/modelpool/MyCustomModelPool/example.yaml
_target_: fusion_bench.modelpool.MyCustomModelPool
_recursive_: False

models:
  _pretrained_: myorg/my-base-model
  task_a: myorg/my-model-finetuned-task-a
  task_b: myorg/my-model-finetuned-task-b
  task_c: myorg/my-model-finetuned-task-c

# Optional: train/val/test datasets
train_datasets: null
val_datasets: null
test_datasets: null

# Custom parameters
custom_processor:
  _target_: transformers.AutoProcessor.from_pretrained
  pretrained_model_name_or_path: myorg/my-base-model

platform: hf
```

Key points:

- **`_target_`**: Must point to the fully qualified Python class path.
- **`_recursive_: False`**: Prevents recursive instantiation of nested configs.
- **`models`**: Dictionary mapping names to model IDs or DictConfig objects. Always include `_pretrained_` when using methods like Task Arithmetic.
- **Special names**: Model names starting and ending with underscores (e.g., `_pretrained_`, `_merged_`) are treated as special models.

## Complete Working Example

Here is a complete, minimal custom ModelPool for a hypothetical BERT-like classifier:

```python
# fusion_bench/modelpool/bert_classifier_pool.py
import logging
from typing import Optional, Union
from typing_extensions import override

from omegaconf import DictConfig
from torch import nn
from transformers import AutoModelForSequenceClassification

from fusion_bench import BaseModelPool

log = logging.getLogger(__name__)


class BertClassifierModelPool(BaseModelPool):
    """Model pool for BERT-based sequence classification models."""

    _config_mapping = BaseModelPool._config_mapping | {
        "_num_labels": "num_labels",
    }

    def __init__(
        self,
        models: DictConfig,
        *,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        self._num_labels = num_labels

    @override
    def load_model(
        self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
    ) -> nn.Module:
        if isinstance(model_name_or_config, str) and model_name_or_config in self._models:
            model_config = self._models[model_name_or_config]

            if isinstance(model_config, str):
                log.info(f"Loading BERT classifier from: {model_config}")
                kwargs.setdefault("num_labels", self._num_labels)
                return AutoModelForSequenceClassification.from_pretrained(
                    model_config, *args, **kwargs
                )

            if isinstance(model_config, nn.Module):
                return model_config

        return super().load_model(model_name_or_config, *args, **kwargs)

    @override
    def save_model(self, model: nn.Module, path: str, *args, **kwargs):
        """Save using HuggingFace format."""
        model.save_pretrained(path)
```

With config file:

```yaml
# config/modelpool/BertClassifierPool/glue_tasks.yaml
_target_: fusion_bench.modelpool.bert_classifier_pool.BertClassifierModelPool
_recursive_: False

models:
  _pretrained_: bert-base-uncased
  sst2: user/bert-sst2-finetuned
  mnli: user/bert-mnli-finetuned

num_labels: 2
```

Usage:

```bash
fusion_bench \
  method=simple_average \
  modelpool=BertClassifierPool/glue_tasks \
  taskpool=dummy
```

## Best Practices

1. **Always call `super().__init__()`**: The base class handles model validation, special name checks, and Hydra integration.
2. **Use `@override` decorator**: From `typing_extensions`, this marks overridden methods and catches errors at runtime.
3. **Handle all three config types**: In `load_model`, handle `str`, `nn.Module`, and `DictConfig` cases. Delegate unexpected types to `super().load_model()`.
4. **Log model loading**: Use `rank_zero_only` logging to avoid duplicate logs in distributed settings.
5. **Support `*args, **kwargs`**: Always forward extra arguments so algorithms can pass device, dtype, or other parameters.
6. **Validate model names**: The base class validates names during `__init__`, but you can add custom validation in `load_model`.

## Next Steps

- See `fusion_bench/modelpool/clip_vision/modelpool.py` and `fusion_bench/modelpool/resnet_for_image_classification.py` for production examples.
- Read the [Custom TaskPool](custom_taskpool.md) guide to create a matching evaluation component.
