# ResNet Models for Image Classification

This page documents the ResNet model pool used in FusionBench for supervised image classification. The pool supports both torchvision and Hugging Face Transformers implementations and provides a unified interface for loading models, processors, and saving artifacts.

## Overview

The [`ResNetForImageClassificationPool`][fusion_bench.modelpool.resnet_for_image_classification.ResNetForImageClassificationPool] offers:

- Two backends
	- `torchvision`: classic ResNet backbones (resnet18/34/50/101/152) with standard ImageNet preprocessing
	- `transformers`: `ResNetForImageClassification` models with `AutoImageProcessor`
- Dataset-aware adaptation
	- When a `dataset_name` is provided, the pool resizes the classifier head and sets `id2label`/`label2id` mappings via the dataset class names
- Processor management
	- Torchvision: returns stage-specific transforms (train/val/test)
	- Transformers: loads a compatible `AutoImageProcessor`
- Clean logits API
	- For Transformers models, `forward` is wrapped to return logits only for consistent evaluation interfaces

## Quick start (Transformers backend)

Minimal Python usage with a single pretrained model adapted to a dataset (e.g., CIFAR-10):

```python
from fusion_bench.modelpool import ResNetForImageClassificationPool

pool = ResNetForImageClassificationPool(
		type="transformers",
		models={
				"_pretrained_": {
						"config_path": "microsoft/resnet-50",
						"pretrained": True,
						"dataset_name": "cifar10",
				}
		},
)

model = pool.load_model("_pretrained_")
processor = pool.load_processor()  # AutoImageProcessor
```

## Torchvision backend

When using torchvision models, the pool constructs appropriate transforms per stage and can optionally resize the classifier to match your dataset:

```python
from fusion_bench.modelpool import ResNetForImageClassificationPool

pool = ResNetForImageClassificationPool(
		type="torchvision",
		models={
				"resnet18_cifar10": {
						"model_name": "resnet18",
						"weights": "DEFAULT",  # or None
						"dataset_name": "cifar10",  # classifier resized to 10 classes
				}
		},
)

train_tf = pool.load_processor(stage="train")
val_tf = pool.load_processor(stage="test")
model = pool.load_model("resnet18_cifar10")
```

Low-level helpers also exist if you want to create models directly:

- [fusion_bench.modelpool.resnet_for_image_classification.load_torchvision_resnet][]
- [fusion_bench.modelpool.resnet_for_image_classification.load_transformers_resnet][]

## Ready-to-use configs

You can use the following model pool configs out-of-the-box. They set up a single pretrained model adapted to a specific dataset. Include them with the `modelpool=` flag when running `fusion_bench`.

```yaml title="config/modelpool/ResNetForImageClassification/transformers/resnet18_cifar10.yaml"
--8<-- "config/modelpool/ResNetForImageClassification/transformers/resnet18_cifar10.yaml"
```

```yaml title="config/modelpool/ResNetForImageClassification/transformers/resnet50_cifar10.yaml"
--8<-- "config/modelpool/ResNetForImageClassification/transformers/resnet50_cifar10.yaml"
```

These configs follow the same structure used across other model pools in this directory (see CLIP-ViT or GPT-2 pages for reference) and are suitable starting points for evaluation and merging workflows.

## Saving models

- Torchvision: `state_dict` is saved via `torch.save` to the given file path
- Transformers: `save_pretrained()` is used for both model and processor; a README model card is written on rank-zero when `algorithm_config` is supplied

See [fusion_bench.modelpool.resnet_for_image_classification.ResNetForImageClassificationPool.save_model][].

## Implementation Details

- Pool class: [fusion_bench.modelpool.resnet_for_image_classification.ResNetForImageClassificationPool][]
- Torchvision loader: [fusion_bench.modelpool.resnet_for_image_classification.load_torchvision_resnet][]
- Transformers loader: [fusion_bench.modelpool.resnet_for_image_classification.load_transformers_resnet][]
- Task pool (evaluation): [fusion_bench.taskpool.resnet_for_image_classification.ResNetForImageClassificationTaskPool][]
