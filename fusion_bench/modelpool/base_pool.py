import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from fusion_bench.mixins import BaseYAMLSerializableModel, HydraConfigMixin
from fusion_bench.utils import instantiate, timeit_context

__all__ = ["BaseModelPool"]

log = logging.getLogger(__name__)


class BaseModelPool(BaseYAMLSerializableModel, HydraConfigMixin):
    """
    A class for managing and interacting with a pool of models along with their associated datasets or other specifications. For example, a model pool may contain multiple models, each with its own training, validation, and testing datasets. As for the specifications, a vision model pool may contain image preprocessor, and a language model pool may contain a tokenizer.

    Attributes:
        _models (DictConfig): Configuration for all models in the pool.
        _train_datasets (Optional[DictConfig]): Configuration for training datasets.
        _val_datasets (Optional[DictConfig]): Configuration for validation datasets.
        _test_datasets (Optional[DictConfig]): Configuration for testing datasets.
        _usage_ (Optional[str]): Optional usage information.
        _version_ (Optional[str]): Optional version information.
    """

    _program = None
    _models: Union[DictConfig, Dict[str, nn.Module]]
    _config_mapping = BaseYAMLSerializableModel._config_mapping | {
        "_models": "models",
        "_train_datasets": "train_datasets",
        "_val_datasets": "val_datasets",
        "_test_datasets": "test_datasets",
    }

    def __init__(
        self,
        models: Union[DictConfig, Dict[str, nn.Module], List[nn.Module]],
        *,
        train_datasets: Optional[DictConfig] = None,
        val_datasets: Optional[DictConfig] = None,
        test_datasets: Optional[DictConfig] = None,
        **kwargs,
    ):
        if isinstance(models, List):
            models = {str(model_idx): model for model_idx, model in enumerate(models)}
        self._models = models
        self._train_datasets = train_datasets
        self._val_datasets = val_datasets
        self._test_datasets = test_datasets
        super().__init__(**kwargs)

    @property
    def has_pretrained(self):
        """
        Check if the model pool contains a pretrained model.

        Returns:
            bool: True if a pretrained model is available, False otherwise.
        """
        return "_pretrained_" in self._models

    @property
    def all_model_names(self) -> List[str]:
        """
        Get the names of all models in the pool, including special models.

        Returns:
            List[str]: A list of all model names.
        """
        return [name for name in self._models]

    @property
    def model_names(self) -> List[str]:
        """
        Get the names of regular models, excluding special models.

        Returns:
            List[str]: A list of regular model names.
        """
        return [name for name in self._models if not self.is_special_model(name)]

    @property
    def train_dataset_names(self) -> List[str]:
        """
        Get the names of training datasets.

        Returns:
            List[str]: A list of training dataset names.
        """
        return (
            list(self._train_datasets.keys())
            if self._train_datasets is not None
            else []
        )

    @property
    def val_dataset_names(self) -> List[str]:
        """
        Get the names of validation datasets.

        Returns:
            List[str]: A list of validation dataset names.
        """
        return list(self._val_datasets.keys()) if self._val_datasets is not None else []

    @property
    def test_dataset_names(self) -> List[str]:
        """
        Get the names of testing datasets.

        Returns:
            List[str]: A list of testing dataset names.
        """
        return (
            list(self._test_datasets.keys()) if self._test_datasets is not None else []
        )

    def __len__(self):
        return len(self.model_names)

    @staticmethod
    def is_special_model(model_name: str):
        """
        Determine if a model is special based on its name.

        Args:
            model_name (str): The name of the model.

        Returns:
            bool: True if the model name indicates a special model, False otherwise.
        """
        return model_name.startswith("_") and model_name.endswith("_")

    def get_model_config(self, model_name: str, return_copy: bool = True) -> DictConfig:
        """
        Get the configuration for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            DictConfig: The configuration for the specified model.
        """
        model_config = self._models[model_name]
        if return_copy:
            model_config = deepcopy(model_config)
        return model_config

    def load_model(
        self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
    ) -> nn.Module:
        """
        Load a model from the pool based on the provided configuration.

        Args:
            model (Union[str, DictConfig]): The model name or configuration.

        Returns:
            nn.Module: The instantiated model.
        """
        log.debug(f"Loading model: {model_name_or_config}", stacklevel=2)
        if isinstance(self._models, DictConfig):
            model_config = (
                self._models[model_name_or_config]
                if isinstance(model_name_or_config, str)
                else model_name_or_config
            )
            model = instantiate(model_config, *args, **kwargs)
        elif isinstance(self._models, Dict) and isinstance(model_name_or_config, str):
            model = self._models[model_name_or_config]
        else:
            raise ValueError(
                "The model pool configuration is not in the expected format."
                f"We expected a DictConfig or Dict, but got {type(self._models)}."
            )
        return model

    def load_pretrained_model(self, *args, **kwargs):
        assert (
            self.has_pretrained
        ), "No pretrained model available. Check `_pretrained_` is in the `models` key."
        model = self.load_model("_pretrained_", *args, **kwargs)
        return model

    def load_pretrained_or_first_model(self, *args, **kwargs):
        """
        Load the pretrained model if available, otherwise load the first available model.

        Returns:
            nn.Module: The loaded model.
        """
        if self.has_pretrained:
            model = self.load_model("_pretrained_", *args, **kwargs)
        else:
            model = self.load_model(self.model_names[0], *args, **kwargs)
        return model

    def models(self):
        for model_name in self.model_names:
            yield self.load_model(model_name)

    def named_models(self):
        for model_name in self.model_names:
            yield model_name, self.load_model(model_name)

    def load_train_dataset(self, dataset_name: str, *args, **kwargs) -> Dataset:
        """
        Load the training dataset for the specified model.

        Args:
            dataset_name (str): The name of the model.

        Returns:
            Dataset: The instantiated training dataset.
        """
        return instantiate(self._train_datasets[dataset_name], *args, **kwargs)

    def train_datasets(self):
        for dataset_name in self.train_dataset_names:
            yield self.load_train_dataset(dataset_name)

    def load_val_dataset(self, dataset_name: str, *args, **kwargs) -> Dataset:
        """
        Load the validation dataset for the specified model.

        Args:
            dataset_name (str): The name of the model.

        Returns:
            Dataset: The instantiated validation dataset.
        """
        return instantiate(self._val_datasets[dataset_name], *args, **kwargs)

    def val_datasets(self):
        for dataset_name in self.val_dataset_names:
            yield self.load_val_dataset(dataset_name)

    def load_test_dataset(self, dataset_name: str, *args, **kwargs) -> Dataset:
        """
        Load the testing dataset for the specified model.

        Args:
            dataset_name (str): The name of the model.

        Returns:
            Dataset: The instantiated testing dataset.
        """
        return instantiate(self._test_datasets[dataset_name], *args, **kwargs)

    def test_datasets(self):
        for dataset_name in self.test_dataset_names:
            yield self.load_test_dataset(dataset_name)

    def save_model(self, model: nn.Module, path: str):
        """
        Save the state dictionary of the model to the specified path.

        Args:
            model (nn.Module): The model whose state dictionary is to be saved.
            path (str): The path where the state dictionary will be saved.
        """
        with timeit_context(f"Saving the state dict of model to {path}"):
            torch.save(model.state_dict(), path)
