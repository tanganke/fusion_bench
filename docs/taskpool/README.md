# Task Pool Module

A taskpool is a collection of tasks that can be used to evaluate the performance of merged models.
Each task in the taskpool is typically associated with a dataset and evaluation metrics.

## Configuration Structure

Starting from version 0.2, taskpools use Hydra-based configuration with the `_target_` field to specify the class to instantiate. A taskpool configuration file typically contains the following fields:

### Core Fields

- `_target_`: The fully qualified class name of the taskpool (e.g., [`fusion_bench.taskpool.CLIPVisionModelTaskPool`][])
- `test_datasets`: A dictionary of test dataset configurations where each key is the task name and the value is the dataset configuration
- Additional model-specific configuration fields (processor, tokenizer, etc.)

### Common Configuration Fields

Different taskpool types may include additional configuration fields:

- `processor`: Configuration for data processors (e.g., image preprocessors, tokenizers)
- `dataloader_kwargs`: Configuration for PyTorch DataLoader (batch_size, num_workers, etc.)
- `fast_dev_run`: Boolean flag for quick development testing
- `base_model`: Base model identifier used for loading processors and other components

## Configuration Examples

### CLIP Vision Model Task Pool

```yaml
_target_: fusion_bench.taskpool.CLIPVisionModelTaskPool
base_model: openai/clip-vit-base-patch32
clip_model:
  _target_: transformers.CLIPModel.from_pretrained
  pretrained_model_name_or_path: ${..base_model}
processor:
  _target_: transformers.CLIPProcessor.from_pretrained
  pretrained_model_name_or_path: ${..base_model}
test_datasets:
  cifar10:
    _target_: datasets.load_dataset
    path: cifar10
    split: test
  cifar100:
    _target_: datasets.load_dataset
    path: cifar100
    split: test
dataloader_kwargs:
  batch_size: 128
  num_workers: 8
```

### GPT-2 Text Classification Task Pool

```yaml
_target_: fusion_bench.taskpool.GPT2TextClassificationTaskPool
test_datasets:
  cola:
    _target_: fusion_bench.taskpool.gpt2_text_classification.load_gpt2_dataset
    name: cola
    split: validation
  sst2:
    _target_: fusion_bench.taskpool.gpt2_text_classification.load_gpt2_dataset
    name: sst2
    split: validation
tokenizer:
  _target_: fusion_bench.modelpool.huggingface_gpt2_classification.load_gpt2_tokenizer
  pretrained_model_name_or_path: gpt2
dataloader_kwargs:
  batch_size: 8
  num_workers: 0
```

### Dummy Task Pool (for debugging)

```yaml
_target_: fusion_bench.taskpool.DummyTaskPool
model_save_path: null
```

## Usage

### Creating a TaskPool

Starting from v0.2, taskpools can be created directly or through Hydra configuration:

```python
# Create from configuration file
from fusion_bench.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load("path/to/taskpool/config.yaml")
taskpool = instantiate(config)

# Create directly
from fusion_bench.taskpool import CLIPVisionModelTaskPool
taskpool = CLIPVisionModelTaskPool(
    test_datasets={
        "cifar10": {
            "_target_": "datasets.load_dataset",
            "path": "cifar10",
            "split": "test"
        }
    },
    processor="openai/clip-vit-base-patch32",
    clip_model="openai/clip-vit-base-patch32"
)
```

### Evaluating Models

The primary function of a taskpool is to evaluate models across multiple tasks:

```python
# Evaluate a model on all tasks in the taskpool
report = taskpool.evaluate(model)

# The report structure:
# {
#     "task_name": {
#         "metric_name": metric_value,
#         ...
#     },
#     ...
# }
```

### Integration with Algorithms

Taskpools can be used within fusion algorithms for evaluation during training:

```python
class CustomAlgorithm(BaseAlgorithm):
    def run(self, modelpool):
        # Your fusion logic here
        merged_model = self.fuse_models(modelpool)
        
        # Evaluate if taskpool is available
        if hasattr(self, '_program') and self._program.taskpool is not None:
            report = self._program.taskpool.evaluate(merged_model)
            print(f"Evaluation results: {report}")
        
        return merged_model
```

### Implementation Details

- [fusion_bench.taskpool.BaseTaskPool][]
