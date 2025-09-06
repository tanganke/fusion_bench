# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from omegaconf import DictConfig, open_dict

from fusion_bench.utils.lazy_imports import LazyImporter


def load_dataset_from_config(dataset_config: DictConfig):
    """
    Load the dataset from the configuration.
    """
    from datasets import load_dataset

    from fusion_bench.utils import instantiate

    assert hasattr(dataset_config, "type"), "Dataset type not specified"
    if dataset_config.type == "instantiate":
        return instantiate(dataset_config.object)
    elif dataset_config.type == "huggingface_image_classification":
        if not hasattr(dataset_config, "path"):
            with open_dict(dataset_config):
                dataset_config.path = dataset_config.name
        dataset = load_dataset(
            dataset_config.path,
            **(dataset_config.kwargs if hasattr(dataset_config, "kwargs") else {}),
        )
        if hasattr(dataset_config, "split"):
            dataset = dataset[dataset_config.split]
        return dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config.type}")


_extra_objects = {
    "load_dataset_from_config": load_dataset_from_config,
}
_import_structure = {
    "clip_dataset": ["CLIPDataset"],
}

if TYPE_CHECKING:
    from .clip_dataset import CLIPDataset

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects=_extra_objects,
    )
