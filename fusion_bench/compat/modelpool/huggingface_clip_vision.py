import functools
import logging
from typing import Optional

from omegaconf import DictConfig, open_dict
from transformers import CLIPProcessor, CLIPVisionModel
from typing_extensions import override

from fusion_bench.dataset import CLIPDataset, load_dataset_from_config
from fusion_bench.utils import timeit_context

from .base_pool import ModelPool

log = logging.getLogger(__name__)


class HuggingFaceClipVisionPool(ModelPool):
    """
    A model pool for managing Hugging Face's CLIP Vision models.

    This class extends the base `ModelPool` class and overrides its methods to handle
    the specifics of the CLIP Vision models provided by the Hugging Face Transformers library.
    """

    def __init__(self, modelpool_config: DictConfig):
        super().__init__(modelpool_config)

        self._clip_processor = None

    @property
    def clip_processor(self):
        """
        Returns the CLIP processor. If it's not already initialized, it initializes it using the path of the pretrained model.
        """
        if self._clip_processor is None:
            if "_pretrained_" in self._model_names:
                self._clip_processor = CLIPProcessor.from_pretrained(
                    self.get_model_config("_pretrained_")["path"]
                )
            else:
                log.warning(
                    "No pretrained model found in the model pool. Returning the first model."
                )
                self._clip_processor = CLIPProcessor.from_pretrained(
                    self.get_model_config(self.model_names[0])["path"]
                )
        return self._clip_processor

    @override
    def load_model(self, model_config: str | DictConfig) -> CLIPVisionModel:
        """
        Load a CLIP Vision model from the given configuration.

        Args:
            model_config (str | DictConfig): The configuration for the model to load.

        Returns:
            CLIPVisionModel: The loaded CLIP Vision model.
        """
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
            f"Loading CLIP vision model: '{model_config.name}' from '{model_config.path}'."
        ):
            vision_model = CLIPVisionModel.from_pretrained(model_config.path)
        return vision_model

    @override
    def save_model(self, model: CLIPVisionModel, path: str):
        """
        Save a CLIP Vision model to the given path.

        Args:
            model (CLIPVisionModel): The model to save.
            path (str): The path to save the model to.
        """
        with timeit_context(f'Saving clip vision model to "{path}"'):
            model.save_pretrained(path)

    def get_tta_dataset_config(self, dataset: str):
        """
        Retrieve the configuration for a TTA (Test-Time Adaptation) dataset.

        Args:
            dataset (str): The name of the dataset for which to retrieve the configuration.

        Returns:
            DictConfig: The configuration dictionary for the specified dataset.

        Raises:
            ValueError: If the specified dataset is not found in the configuration.
        """
        for dataset_config in self.config.tta_datasets:
            if dataset_config.name == dataset:
                return dataset_config
        raise ValueError(f"Dataset {dataset} not found in config")

    def prepare_dataset_config(self, dataset_config: DictConfig):
        """
        Prepare the dataset configuration by setting the dataset type if it's not already set.

        Args:
            dataset_config (DictConfig): The configuration dictionary for the dataset.

        Returns:
            DictConfig: The updated configuration dictionary for the dataset.
        """
        if not hasattr(dataset_config, "type"):
            with open_dict(dataset_config):
                dataset_config["type"] = self.config.dataset_type
        return dataset_config

    @functools.cache
    def get_tta_test_dataset(
        self, tta_dataset: str, clip_processor: Optional[CLIPProcessor] = None
    ):
        """
        Load the test dataset for the task.
        This method is cached, so the dataset is loaded only once.

        Args:
            tta_dataset (str): The name of the TTA dataset to load.
            clip_processor (Optional[CLIPProcessor]): The CLIP processor to use for preprocessing the dataset. If None, the default processor is used.

        Returns:
            CLIPDataset: The loaded and preprocessed TTA test dataset.
        """
        if clip_processor is None:
            # if clip_processor is not provided, try to load the clip_processor from pre-trained model
            clip_processor = self.clip_processor
        dataset_config = self.get_tta_dataset_config(tta_dataset)["dataset"]
        dataset_config = self.prepare_dataset_config(dataset_config)
        with timeit_context(f"Loading test dataset: {dataset_config.name}"):
            dataset = load_dataset_from_config(dataset_config)
        dataset = CLIPDataset(dataset, self.clip_processor)
        return dataset

    def get_train_dataset_config(self, model_name: str):
        """
        Retrieve the configuration for a specific training dataset.

        Args:
            model_name (str): The name of the model for which to retrieve the training dataset configuration.

        Returns:
            DictConfig: The configuration dictionary for the specified training dataset.

        Raises:
            ValueError: If the specified training dataset is not found in the configuration.
        """
        for dataset_config in self.config.train_datasets:
            if dataset_config.name == model_name:
                return dataset_config
        raise ValueError(f"Dataset {model_name} not found in config")

    def get_train_dataset(
        self, model_name: str, clip_processor: Optional[CLIPProcessor] = None
    ):
        """
        Load the training dataset for the specified model.

        Args:
            model_name (str): The name of the model for which to load the training dataset.
            clip_processor (Optional[CLIPProcessor]): The CLIP processor to use for preprocessing the dataset. If None, the default processor is used.

        Returns:
            CLIPDataset: The loaded and preprocessed training dataset.
        """
        if clip_processor is None:
            # if clip_processor is not provided, try to load the clip_processor from pre-trained model
            clip_processor = self.clip_processor
        dataset_config = self.get_train_dataset_config(model_name)["dataset"]
        dataset_config = self.prepare_dataset_config(dataset_config)
        with timeit_context(f"Loading train dataset: {dataset_config.name}"):
            dataset = load_dataset_from_config(dataset_config)
        dataset = CLIPDataset(dataset, self.clip_processor)
        return dataset
