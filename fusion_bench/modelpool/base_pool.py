import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from fusion_bench.mixins import YAMLSerializationMixin
from fusion_bench.utils import timeit_context

__all__ = ["ModelPool", "DictModelPool", "ListModelPool", "to_modelpool"]

log = logging.getLogger(__name__)


class ModelPool(ABC):
    """
    This is the base class for all modelpools.
    """

    _model_names = None

    def __init__(self, modelpool_config: Optional[DictConfig] = None):
        super().__init__()
        self.config = modelpool_config

        # check for duplicate model names
        if self.config is not None and self.config.get("models", None) is not None:
            model_names = [model["name"] for model in self.config["models"]]
            assert len(model_names) == len(
                set(model_names)
            ), "Duplicate model names found in model pool"
            self._model_names = model_names

    def __len__(self):
        return len(self.model_names)

    @property
    def model_names(self) -> List[str]:
        """
        This property returns a list of model names from the configuration, excluding any names that start or end with an underscore.
        To obtain all model names, including those starting or ending with an underscore, use the `_model_names` attribute.

        Returns:
            list: A list of model names.
        """
        names = [
            name for name in self._model_names if name[0] != "_" and name[-1] != "_"
        ]
        return names

    @property
    def has_pretrained(self):
        """
        Check if the pretrained model is available in the model pool.
        """
        for model_config in self.config["models"]:
            if model_config.get("name", None) == "_pretrained_":
                return True
        return False

    def get_model_config(self, model_name: str):
        """
        Retrieves the configuration for a specific model from the model pool.

        Args:
            model_name (str): The name of the model for which to retrieve the configuration.

        Returns:
            dict: The configuration dictionary for the specified model.

        Raises:
            ValueError: If the specified model is not found in the model pool.
        """
        for model in self.config["models"]:
            if model["name"] == model_name:
                return model
        raise ValueError(f"Model {model_name} not found in model pool")

    def load_model(self, model_config: Union[str, DictConfig]) -> nn.Module:
        """
        The models are load lazily, so this method should be implemented to load the model from the model pool.

        Load the model from the model pool.

        Args:
            model_config (Union[str, DictConfig]): The configuration dictionary for the model to load.

        Returns:
            Any: The loaded model.
        """
        raise NotImplementedError

    def load_pretrained_or_first_model(self, *args, **kwargs):
        """
        Load the pretrained model if available, otherwise load the first model in the list.

        This method checks if a pretrained model is available. If it is, it loads the pretrained model.
        If not, it loads the first model from the list of model names.

        Returns:
            nn.Module: The loaded model.
        """
        if self.has_pretrained:
            model = self.load_model("_pretrained_", *args, **kwargs)
        else:
            model = self.load_model(self.model_names[0], *args, **kwargs)
        return model

    def save_model(self, model: nn.Module, path: str):
        """
        Save the state dictionary of the model to the specified path.

        Args:
            model (nn.Module): The model whose state dictionary is to be saved.
            path (str): The path where the state dictionary will be saved.
        """
        with timeit_context(f"Saving the state dict of model to {path}"):
            torch.save(model.state_dict(), path)

    def models(self):
        for model_name in self.model_names:
            yield self.load_model(model_name)

    def named_models(self):
        for model_name in self.model_names:
            yield model_name, self.load_model(model_name)

    def get_train_dataset(self, model_name: str):
        """
        Get the training dataset for the model.
        """
        raise NotImplementedError

    def get_test_dataset(self, model_name: str):
        """
        Get the testing dataset for the model.
        """
        raise NotImplementedError

    def setup_taskpool(self, taskpool):
        """
        Setup the taskpool before evaluation.
        Such as setting the fabric, processor, tokenizer, etc.
        """
        pass

    def to_modellist(self) -> List[nn.Module]:
        """
        Convert the model pool to a list of models.
        """
        return [self.load_model(m) for m in self.model_names]

    def to_modeldict(self) -> Dict[str, nn.Module]:
        """
        Convert the model pool to a dictionary of models.
        """
        return {m: self.load_model(m) for m in self.model_names}


class ListModelPool(ModelPool):
    """
    ModelPool from a list of models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        has_pretraned: bool = False,
    ):
        modelpool_config = {}
        modelpool_config["models"] = []
        model_dict = {}
        if has_pretraned:
            model_dict["_pretrained_"] = models[0]
            modelpool_config["models"].append({"name": "_pretrained_"})
            for i, model in enumerate(models[1:]):
                model_dict[f"model_{i}"] = model
                modelpool_config["models"].append({"name": f"model_{i}"})
        else:
            for i, model in enumerate(models):
                model_dict[f"model_{i}"] = model
                modelpool_config["models"].append({"name": f"model_{i}"})

        self.model_dict = model_dict
        super().__init__(DictConfig(modelpool_config))

    def load_model(self, model_config: str | DictConfig, copy=True) -> nn.Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        model_name = model_config["name"]
        model = self.model_dict[model_name]
        if copy:
            model = deepcopy(model)
        return model


class DictModelPool(ModelPool):
    """
    ModelPool from a dictionary of models.
    """

    def __init__(self, model_dict: Dict[str, nn.Module]):
        modelpool_config = {}
        modelpool_config["models"] = []
        for model_name, model in model_dict.items():
            modelpool_config["models"].append({"name": model_name})
        self.model_dict = model_dict
        super().__init__(DictConfig(modelpool_config))

    def load_model(self, model_config: str | DictConfig, copy=True) -> nn.Module:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)
        model_name = model_config["name"]
        model = self.model_dict[model_name]
        if copy:
            model = deepcopy(model)
        return model


def to_modelpool(obj: List[nn.Module], **kwargs):
    if isinstance(obj, ModelPool):
        return obj
    elif isinstance(obj, (list, tuple)) and all(isinstance(m, nn.Module) for m in obj):
        return ListModelPool(models=obj, **kwargs)
    elif isinstance(obj, Dict) and all(isinstance(m, nn.Module) for m in obj.values()):
        return DictModelPool(model_dict=obj, **kwargs)
    elif isinstance(obj, nn.Module):
        return ListModelPool(models=[obj], **kwargs)
    else:
        raise ValueError(f"Invalid modelpool object: {obj}")


class BaseModelPool(YAMLSerializationMixin):
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

    _models: DictConfig
    _config_mapping = YAMLSerializationMixin._config_mapping | {
        "_models": "models",
        "_train_datasets": "train_datasets",
        "_val_datasets": "val_datasets",
        "_test_datasets": "test_datasets",
        "_usage_": "_usage_",
        "_version_": "_version_",
    }

    def __init__(
        self,
        models: DictConfig,
        *,
        train_datasets: Optional[DictConfig] = None,
        val_datasets: Optional[DictConfig] = None,
        test_datasets: Optional[DictConfig] = None,
        _usage_: Optional[str] = None,
        _version_: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._models = models
        self._train_datasets = train_datasets
        self._val_datasets = val_datasets
        self._test_datasets = test_datasets
        self._usage_ = _usage_
        self._version_ = _version_
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def has_pretrained(self):
        """
        Check if the model pool contains a pretrained model.

        Returns:
            bool: True if a pretrained model is available, False otherwise.
        """
        return "_pretrained_" in self._models

    @property
    def all_model_names(self):
        """
        Get the names of all models in the pool, including special models.

        Returns:
            List[str]: A list of all model names.
        """
        return self._models

    @property
    def model_names(self):
        """
        Get the names of regular models, excluding special models.

        Returns:
            List[str]: A list of regular model names.
        """
        return [name for name in self._models if not self.is_special_model(name)]

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

    def load_model(self, model: Union[str, DictConfig], *args, **kwargs) -> nn.Module:
        """
        Load a model from the pool based on the provided configuration.

        Args:
            model (Union[str, DictConfig]): The model name or configuration.

        Returns:
            nn.Module: The instantiated model.
        """
        if isinstance(model, str):
            model = self._models[model]
        return instantiate(model, *args, **kwargs)

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

    def load_train_dataset(self, model_name: str, *args, **kwargs) -> Dataset:
        """
        Load the training dataset for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Dataset: The instantiated training dataset.
        """
        return instantiate(self._train_datasets[model_name], *args, **kwargs)

    def load_val_dataset(self, model_name: str, *args, **kwargs) -> Dataset:
        """
        Load the validation dataset for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Dataset: The instantiated validation dataset.
        """
        return instantiate(self._val_datasets[model_name], *args, **kwargs)

    def load_test_dataset(self, model_name: str, *args, **kwargs) -> Dataset:
        """
        Load the testing dataset for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Dataset: The instantiated testing dataset.
        """
        return instantiate(self._test_datasets[model_name], *args, **kwargs)

    def save_model(self, model: nn.Module, path: str):
        """
        Save the state dictionary of the model to the specified path.

        Args:
            model (nn.Module): The model whose state dictionary is to be saved.
            path (str): The path where the state dictionary will be saved.
        """
        with timeit_context(f"Saving the state dict of model to {path}"):
            torch.save(model.state_dict(), path)
