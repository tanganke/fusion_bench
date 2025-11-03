# DINOv2 Models for Image Classification

This page documents the DINOv2 image classification model pool in FusionBench. It wraps Hugging Face Transformers DINOv2 models with convenient loading, processor management, dataset-aware head adaptation, and save utilities.

Implementation: [Dinov2ForImageClassificationPool][fusion_bench.modelpool.dinov2_for_image_classification.Dinov2ForImageClassificationPool]

## Quick start

Minimal Python usage with a single pretrained DINOv2 model (e.g., base imagenet1k 1-layer):

```python
from fusion_bench.modelpool import Dinov2ForImageClassificationPool

pool = Dinov2ForImageClassificationPool(
	models={
		"_pretrained_": {
			"config_path": "facebook/dinov2-base-imagenet1k-1-layer",
			"pretrained": True,
			# set to a known dataset key (e.g., "cifar10") to resize classifier
			# and populate id2label/label2id mappings
			"dataset_name": None,
		}
	}
)

model = pool.load_model("_pretrained_")
processor = pool.load_processor()  # AutoImageProcessor
```

Low-level construction is available via helpers:

- [fusion_bench.modelpool.dinov2_for_image_classification.load_transformers_dinov2][]

## Ready-to-use config

Use the provided Hydra config to set up a pretrained DINOv2-base (imagenet1k 1-layer) model:

```yaml title="config/modelpool/Dinov2ForImageClassification/dinov2-base-imagenet1k-1-layer.yaml"
--8<-- "config/modelpool/Dinov2ForImageClassification/dinov2-base-imagenet1k-1-layer.yaml"
```

Tip: set `dataset_name` to a supported dataset key (e.g., `cifar10`, `svhn`, `gtsrb`, …) to auto-resize the classifier and label mappings.

## Implementation Details

- Pool class: [fusion_bench.modelpool.dinov2_for_image_classification.Dinov2ForImageClassificationPool][]
- Loader helper: [fusion_bench.modelpool.dinov2_for_image_classification.load_transformers_dinov2][]

