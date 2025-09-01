# CLIP Template Factory Documentation

## Overview

[`CLIPTemplateFactory`](https://github.com/tanganke/fusion_bench/blob/main/fusion_bench/tasks/clip_classification/__init__.py) is a class designed to facilitate the dynamic creation and management of dataset templates for use with CLIP models. It serves as a factory class that allows users to retrieve class names and templates for various datasets, register new datasets, and obtain a list of all available datasets.

## Usage Example

```python
from fusion_bench.tasks.clip_classification import CLIPTemplateFactory

# List all available datasets
available_datasets = CLIPTemplateFactory.get_available_datasets()
print(available_datasets)
```

get class names and templates for image classification

```python
classnames, templates = CLIPTemplateFactory.get_classnames_and_templates("cifar10")
# classnames: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# templates is a list functions, `templates[0](classnames[0])` will return 'a photo of a airplane.'

# or you can use the `get_classnames_and_templates` function
from fusion_bench.tasks.clip_classification import get_classnames_and_templates

classnames, templates = get_classnames_and_templates("cifar10")
```

or you can register a new dataset

```python
CLIPTemplateFactory.register_dataset(
    "new_dataset",
    dataset_info={
        "module": "module_name",
        "classnames": "classnames",
        "templates": "templates"
    }
)
# Retrieve class names and templates for a registered dataset
# this is equivalent to:
# >>> from module_name import classnames, templates
classnames, templates = CLIPTemplateFactory.get_classnames_and_templates("new_dataset")

# or pass the classnames and templates directly
CLIPTemplaetFactory.register_dataset(
    "new_dataset",
    classnames=["class1", "class2", "class3"],
    templates=[
        lambda x: f"a photo of a {x}.",
        lambda x: f"a picture of a {x}.",
        lambda x: f"an image of a {x}."
    ]
)
```

## Reference

For detailed API documentation, see [fusion_bench.tasks.clip_classification.CLIPTemplateFactory][fusion_bench.tasks.clip_classification.CLIPTemplateFactory] in the API reference.

