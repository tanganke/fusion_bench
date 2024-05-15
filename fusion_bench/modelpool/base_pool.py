from abc import ABC, abstractmethod

from omegaconf import DictConfig
from typing import Union, List


class ModelPool(ABC):
    models = {}

    def __init__(self, modelpool_config: DictConfig):
        super().__init__()
        self.config = modelpool_config

        # check for duplicate model names
        model_names = [model["name"] for model in self.config["models"]]
        assert len(model_names) == len(set(model_names))
        self._model_names = model_names

    @property
    def model_names(self) -> List[str]:
        """
        This property returns a list of model names from the configuration, excluding any names that start or end with an underscore.

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
        for model in self.config["models"]:
            if "pretrained" in model:
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

    @abstractmethod
    def load_model(self, model_config: Union[str, DictConfig]):
        """
        Load the model from the model pool.

        Args:
            model_config (Union[str, DictConfig]): The configuration dictionary for the model to load.

        Returns:
            Any: The loaded model.
        """
        raise NotImplementedError
