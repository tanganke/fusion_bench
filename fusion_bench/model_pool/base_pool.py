from abc import ABC, abstractmethod

from omegaconf import DictConfig


class BasePool(ABC):
    models = {}

    def __init__(self, modelpool_config: DictConfig):
        super().__init__()
        self.config = modelpool_config

        # check for duplicate model names
        model_names = [model["name"] for model in self.config["models"]]
        assert len(model_names) == len(set(model_names))
        self.model_names = model_names

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
