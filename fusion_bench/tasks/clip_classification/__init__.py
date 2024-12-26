import importlib
import warnings
from typing import Any, Callable, Dict, List

from datasets import load_dataset

from .clip_dataset import CLIPDataset


def _check_module_name(module_name: str):
    assert isinstance(
        module_name, str
    ), f"module_name must be a string, got {module_name}"
    if module_name.startswith("."):
        module_name = f"fusion_bench.tasks.clip_classification.{module_name[1:]}"
    return module_name


class CLIPTemplateFactory:
    """
    A factory class for creating CLIP dataset templates.

    This class provides methods to retrieve class names and templates for various datasets,
    register new datasets, and get a list of all available datasets. It uses a mapping
    from dataset names to their respective module paths or detailed information, facilitating
    dynamic import and usage of dataset-specific class names and templates.

    Attributes:
        _dataset_mapping (dict): A mapping from dataset names to their respective module paths
        or detailed information including module path, class names, and templates.

    Methods:
        get_classnames_and_templates(dataset_name: str): Retrieves class names and templates for the specified dataset.
        register_dataset(dataset_name: str, dataset_info: Dict[str, Any] = None, classnames: List[str] = None, templates: List[Callable] = None): Registers a new dataset with its associated information.
        get_available_datasets(): Returns a list of all available dataset names.
    """

    _dataset_mapping = {
        "mnist": ".mnist",
        "stanford-cars": ".stanford_cars",
        "stanford_cars": ".stanford_cars",
        "tanganke/stanford_cars": ".stanford_cars",
        "gtsrb": ".gtsrb",
        "tanganke/gtsrb": ".gtsrb",
        "resisc45": ".resisc45",
        "tanganke/resisc45": ".resisc45",
        "dtd": ".dtd",
        "tanganke/dtd": ".dtd",
        "eurosat": ".eurosat",
        "tanganke/eurosat": ".eurosat",
        "sun397": ".sun397",
        "tanganke/sun397": ".sun397",
        "cifar10": ".cifar10",
        "svhn": ".svhn",
        "cifar100": {
            "module": ".cifar100",
            "classnames": "fine_label",
            "templates": "templates",
        },
        "nateraw/rendered-sst2": ".rendered_sst2",
        "rendered-sst2": ".rendered_sst2",
        "tanganke/stl10": ".stl10",
        "stl10": ".stl10",
        "dpdl-benchmark/oxford_flowers102": ".flower102",
        "oxford_flowers102": ".flower102",
        "timm/oxford-iiit-pet": ".oxford_iiit_pet",
        "oxford-iiit-pet": ".oxford_iiit_pet",
        "imagenet": ".imagenet",
        "tiny-imagenet": ".tiny_imagenet",
        "pcam": ".pcam",
        "fer2013": ".fer2013",
        "emnist_mnist": ".emnist_mnist",
        "emnist_letters": ".emnist_letters",
        "kmnist": ".kmnist",
        "food101": ".food101",
        "fashion_mnist": ".fashion_mnist",
        "cub-200-2011": ".cub_200_2011",
        "mango-leaf-disease": ".mango_leaf_disease",
    }

    @staticmethod
    def get_classnames_and_templates(dataset_name: str):
        """
        Retrieves class names and templates for the specified dataset.

        This method looks up the dataset information in the internal mapping and dynamically imports
        the class names and templates from the specified module. It supports both simple string mappings
        and detailed dictionary mappings for datasets.

        Args:
            dataset_name (str): The name of the dataset for which to retrieve class names and templates.

        Returns:
            Tuple[List[str], List[Callable]]: A tuple containing a list of class names and a list of template callables.

        Raises:
            ValueError: If the dataset_name is not found in the internal mapping.
        """
        if dataset_name not in CLIPTemplateFactory._dataset_mapping:
            raise ValueError(
                f"Unknown dataset {dataset_name}, available datasets: {CLIPTemplateFactory._dataset_mapping.keys()}. You can register a new dataset using `CLIPTemplateFactory.register_dataset()` method."
            )

        dataset_info = CLIPTemplateFactory._dataset_mapping[dataset_name]
        # convert dataset_info to dict format: { 'module': str, 'classnames': str, 'templates': str }
        if isinstance(dataset_info, str):
            dataset_info = _check_module_name(dataset_info)
            dataset_info = {
                "module": dataset_info,
                "classnames": "classnames",
                "templates": "templates",
            }
        elif isinstance(dataset_info, dict):
            if "module" in dataset_info:
                dataset_info["module"] = _check_module_name(dataset_info["module"])

        # import classnames and templates from the specified module
        # convert to dict format: { 'labels': List[str], 'templates': List[Callable] }
        if "module" in dataset_info:
            module = importlib.import_module(dataset_info["module"])
            classnames = getattr(module, dataset_info["classnames"])
            templates = getattr(module, dataset_info["templates"])
        else:
            classnames = dataset_info["classnames"]
            templates = dataset_info["templates"]

        return classnames, templates

    @staticmethod
    def register_dataset(
        dataset_name: str,
        *,
        dataset_info: Dict[str, Any] = None,
        classnames: List[str] = None,
        templates: List[Callable] = None,
    ):
        """
        Registers a new dataset with its associated information.

        This method allows for the dynamic addition of datasets to the internal mapping. It supports
        registration through either a detailed dictionary (`dataset_info`) or separate lists of class names
        and templates. If a dataset with the same name already exists, it will be overwritten.

        The expected format and contents of `dataset_info` can vary depending on the needs of the dataset being registered, but typically, it includes the following keys:

        - "module": A string specifying the module path where the dataset's related classes and functions are located. This is used for dynamic import of the dataset's class names and templates.
        - "classnames": This key is expected to hold the name of the attribute or variable in the specified module that contains a list of class names relevant to the dataset. These class names are used to label data points in the dataset.
        - "templates": Similar to "classnames", this key should specify the name of the attribute or variable in the module that contains a list of template callables. These templates are functions or methods that define how data points should be processed or transformed.

        Args:
            dataset_name (str): The name of the dataset to register.
            dataset_info (Dict[str, Any], optional): A dictionary containing the dataset information, including module path, class names, and templates. Defaults to None.
            classnames (List[str], optional): A list of class names for the dataset. Required if `dataset_info` is not provided. Defaults to None.
            templates (List[Callable], optional): A list of template callables for the dataset. Required if `dataset_info` is not provided. Defaults to None.

        Raises:
            AssertionError: If neither `dataset_info` nor both `classnames` and `templates` are provided.
        """
        assert dataset_info is None or (
            classnames is not None and templates is not None
        ), "You must provide either `dataset_info` or both `classnames` and `templates`."

        if dataset_name in CLIPTemplateFactory._dataset_mapping:
            warnings.warn(
                f"Dataset {dataset_name} is already registered, overwriting the existing dataset information."
            )
        if dataset_info is None:
            dataset_info = {"classnames": classnames, "temolates": templates}
        CLIPTemplateFactory._dataset_mapping[dataset_name] = dataset_info

    @staticmethod
    def get_available_datasets():
        """
        Get a list of all available dataset names.

        Returns:
            List[str]: A list of dataset names.
        """
        return list(CLIPTemplateFactory._dataset_mapping.keys())


def get_classnames_and_templates(dataset_name: str):
    return CLIPTemplateFactory.get_classnames_and_templates(dataset_name)
