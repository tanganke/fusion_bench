# ConvNeXt Models for Image Classification

This page documents the ConvNeXt image classification model pool in FusionBench. It wraps Hugging Face Transformers ConvNeXt models with convenient loading, processor management, dataset-aware head adaptation, and save utilities.

Implementation: [ConvNextForImageClassificationPool][fusion_bench.modelpool.convnext_for_image_classification.ConvNextForImageClassificationPool]

![alt text](images/convnext_block.png)

## Quick start

Minimal Python usage with a single pretrained ConvNeXt model (e.g., base-224):

```python
from fusion_bench.modelpool import ConvNextForImageClassificationPool

pool = ConvNextForImageClassificationPool(
		models={
				"_pretrained_": {
						"config_path": "facebook/convnext-base-224",
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

- [fusion_bench.modelpool.convnext_for_image_classification.load_transformers_convnext][]

## Ready-to-use config

Use the provided Hydra config to set up a pretrained ConvNeXt-base model:

```yaml title="config/modelpool/ConvNextForImageClassification/convnext-base-224.yaml"
--8<-- "config/modelpool/ConvNextForImageClassification/convnext-base-224.yaml"
```

Tip: set `dataset_name` to a supported dataset key (e.g., `cifar10`, `svhn`, `gtsrb`, …) to auto-resize the classifier and label mappings.

## Implementation Details

- Pool class: [fusion_bench.modelpool.convnext_for_image_classification.ConvNextForImageClassificationPool][]
- Loader helper: [fusion_bench.modelpool.convnext_for_image_classification.load_transformers_convnext][]
