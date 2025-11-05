import logging
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf, UnsupportedValueType
from torch import nn
from torch.utils.data import Dataset

from fusion_bench.mixins import BaseYAMLSerializable, HydraConfigMixin
from fusion_bench.utils import (
    ValidationError,
    instantiate,
    timeit_context,
    validate_model_name,
)

__all__ = ["BaseModelPool"]

log = logging.getLogger(__name__)


class BaseModelPool(
    HydraConfigMixin,
    BaseYAMLSerializable,
):
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
    _config_key = "modelpool"
    _models: Union[DictConfig, Dict[str, nn.Module]]
    _config_mapping = BaseYAMLSerializable._config_mapping | {
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

        if isinstance(models, dict):
            try:  # try to convert to DictConfig
                models = OmegaConf.create(models)
            except UnsupportedValueType:
                pass

        if not models:
            log.warning("Initialized BaseModelPool with empty models dictionary.")
        else:
            # Validate model names
            for model_name in models.keys():
                try:
                    validate_model_name(model_name, allow_special=True)
                except ValidationError as e:
                    log.warning(f"Invalid model name '{model_name}': {e}")

        self._models = models
        self._train_datasets = train_datasets
        self._val_datasets = val_datasets
        self._test_datasets = test_datasets
        super().__init__(**kwargs)

    @property
    def has_pretrained(self) -> bool:
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
    def is_special_model(model_name: str) -> bool:
        """
        Determine if a model is special based on its name.

        Args:
            model_name (str): The name of the model.

        Returns:
            bool: True if the model name indicates a special model, False otherwise.
        """
        return model_name.startswith("_") and model_name.endswith("_")

    def get_model_config(
        self, model_name: str, return_copy: bool = True
    ) -> Union[DictConfig, str, Any]:
        """
        Get the configuration for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Union[DictConfig, str, Any]: The configuration for the specified model, which may be a DictConfig, string path, or other type.

        Raises:
            ValidationError: If model_name is invalid.
            KeyError: If model_name is not found in the pool.
        """
        # Validate model name
        validate_model_name(model_name, allow_special=True)

        # raise friendly error if model not found in the pool
        if model_name not in self._models:
            available_models = list(self._models.keys())
            raise KeyError(
                f"Model '{model_name}' not found in model pool. "
                f"Available models: {available_models}"
            )

        model_config = self._models[model_name]
        if isinstance(model_config, nn.Module):
            log.warning(
                f"Model configuration for '{model_name}' is a pre-instantiated model. "
                "Returning the model instance instead of configuration."
            )

        if return_copy:
            if isinstance(model_config, nn.Module):
                # raise performance warning
                log.warning(
                    f"Furthermore, returning a copy of the pre-instantiated model '{model_name}' may be inefficient."
                )
            model_config = deepcopy(model_config)
        return model_config

    def get_model_path(self, model_name: str) -> str:
        """
        Get the path for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            str: The path for the specified model.

        Raises:
            ValidationError: If model_name is invalid.
            KeyError: If model_name is not found in the pool.
            ValueError: If model configuration is not a string path.
        """
        # Validate model name
        validate_model_name(model_name, allow_special=True)

        if model_name not in self._models:
            available_models = list(self._models.keys())
            raise KeyError(
                f"Model '{model_name}' not found in model pool. "
                f"Available models: {available_models}"
            )

        if isinstance(self._models[model_name], str):
            return self._models[model_name]
        else:
            raise ValueError(
                f"Model configuration for '{model_name}' is not a string path. "
                "Try to override this method in derived modelpool class."
            )

    def load_model(
        self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
    ) -> nn.Module:
        """
        Load a model from the pool based on the provided configuration.

        Args:
            model_name_or_config (Union[str, DictConfig]): The model name or configuration.
                - If str: should be a key in self._models
                - If DictConfig: should be a configuration dict for instantiation
            *args: Additional positional arguments passed to model instantiation.
            **kwargs: Additional keyword arguments passed to model instantiation.

        Returns:
            nn.Module: The instantiated or retrieved model.
        """
        log.debug(f"Loading model: {model_name_or_config}", stacklevel=2)

        if isinstance(model_name_or_config, str):
            model_name = model_name_or_config
            # Handle string model names - lookup in the model pool
            if model_name not in self._models:
                raise KeyError(
                    f"Model '{model_name}' not found in model pool. "
                    f"Available models: {list(self._models.keys())}"
                )
            model_config = self._models[model_name]

            # Handle different types of model configurations
            match model_config:
                case dict() | DictConfig() as config:
                    # Configuration that needs instantiation
                    log.debug(f"Instantiating model '{model_name}' from configuration")
                    return instantiate(config, *args, **kwargs)

                case nn.Module() as model:
                    # Pre-instantiated model - return directly
                    log.debug(
                        f"Returning pre-instantiated model '{model_name}' of type {type(model)}"
                    )
                    return model

                case _:
                    # Unsupported model configuration type
                    raise ValueError(
                        f"Unsupported model configuration type for '{model_name}': {type(model_config)}. "
                        f"Expected nn.Module, dict, or DictConfig."
                    )

        elif isinstance(model_name_or_config, (dict, DictConfig)):
            # Direct configuration - instantiate directly
            log.debug("Instantiating model from direct DictConfig")
            model_config = model_name_or_config
            return instantiate(model_config, *args, **kwargs)

        else:
            # Unsupported input type
            raise TypeError(
                f"Unsupported input type: {type(model_name_or_config)}. "
                f"Expected str or DictConfig."
            )

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

    def models(self) -> Generator[nn.Module, None, None]:
        for model_name in self.model_names:
            yield self.load_model(model_name)

    def named_models(self) -> Generator[Tuple[str, nn.Module], None, None]:
        for model_name in self.model_names:
            yield model_name, self.load_model(model_name)

    @property
    def has_train_dataset(self) -> bool:
        """
        Check if the model pool contains training datasets.

        Returns:
            bool: True if training datasets are available, False otherwise.
        """
        return self._train_datasets is not None and len(self._train_datasets) > 0

    @property
    def has_val_dataset(self) -> bool:
        """
        Check if the model pool contains validation datasets.

        Returns:
            bool: True if validation datasets are available, False otherwise.
        """
        return self._val_datasets is not None and len(self._val_datasets) > 0

    @property
    def has_test_dataset(self) -> bool:
        """
        Check if the model pool contains testing datasets.

        Returns:
            bool: True if testing datasets are available, False otherwise.
        """
        return self._test_datasets is not None and len(self._test_datasets) > 0

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

    def save_model(self, model: nn.Module, path: str, *args, **kwargs):
        """
        Save the state dictionary of the model to the specified path.

        Args:
            model (nn.Module): The model whose state dictionary is to be saved.
            path (str): The path where the state dictionary will be saved.
        """
        with timeit_context(f"Saving the state dict of model to {path}"):
            torch.save(model.state_dict(), path)

    def __contains__(self, model_name: str) -> bool:
        """
        Check if a model with the given name exists in the model pool.

        Examples:
            >>> modelpool = BaseModelPool(models={"modelA": ..., "modelB": ...})
            >>> "modelA" in modelpool
            True
            >>> "modelC" in modelpool
            False

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        if self._models is None:
            raise RuntimeError("Model pool is not initialized")
        validate_model_name(model_name, allow_special=True)
        return model_name in self._models
