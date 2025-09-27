import logging
from copy import deepcopy
from typing import Literal, Optional, Union

from datasets import load_dataset
from lightning.fabric.utilities import rank_zero_only
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from typing_extensions import override

from fusion_bench.utils import instantiate, timeit_context
from fusion_bench.utils.modelscope import resolve_repo_path

from ..base_pool import BaseModelPool

log = logging.getLogger(__name__)


class CLIPVisionModelPool(BaseModelPool):
    """
    A model pool for managing Hugging Face's CLIP Vision models.

    This class extends the base `ModelPool` class and overrides its methods to handle
    the specifics of the CLIP Vision models provided by the Hugging Face Transformers library.
    """

    _config_mapping = BaseModelPool._config_mapping | {
        "_processor": "processor",
        "_platform": "hf",
    }

    def __init__(
        self,
        models: DictConfig,
        *,
        processor: Optional[DictConfig] = None,
        platform: Literal["hf", "huggingface", "modelscope"] = "hf",
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        self._processor = processor
        self._platform = platform

    def load_processor(self, *args, **kwargs) -> CLIPProcessor:
        assert self._processor is not None, "Processor is not defined in the config"
        if isinstance(self._processor, str):
            if rank_zero_only.rank == 0:
                log.info(f"Loading `transformers.CLIPProcessor`: {self._processor}")
            repo_path = resolve_repo_path(
                repo_id=self._processor, repo_type="model", platform=self._platform
            )
            processor = CLIPProcessor.from_pretrained(repo_path, *args, **kwargs)
        else:
            processor = instantiate(self._processor, *args, **kwargs)
        return processor

    def load_clip_model(self, model_name: str, *args, **kwargs) -> CLIPModel:
        model_config = self._models[model_name]

        if isinstance(model_config, str):
            if rank_zero_only.rank == 0:
                log.info(f"Loading `transformers.CLIPModel`: {model_config}")
            repo_path = resolve_repo_path(
                repo_id=model_config, repo_type="model", platform=self._platform
            )
            clip_model = CLIPModel.from_pretrained(repo_path, *args, **kwargs)
            return clip_model
        else:
            assert isinstance(
                model_config, DictConfig
            ), "Model config must be a DictConfig"
            model_config = deepcopy(model_config)
            with open_dict(model_config):
                model_config._target_ = "transformers.CLIPModel.from_pretrained"
            clip_model = instantiate(model_config, *args, **kwargs)
            return clip_model

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

    def load_model(
        self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
    ) -> CLIPVisionModel:
        """
        Load a CLIPVisionModel from the model pool with support for various configuration formats.

        This method provides flexible model loading capabilities, handling different types of model
        configurations including string paths, pre-instantiated models, and complex configurations.

        Supported configuration formats:
        1. String model paths (e.g., Hugging Face model IDs)
        2. Pre-instantiated nn.Module objects
        3. DictConfig objects for complex configurations

        Example configuration:
        ```yaml
        models:
            # Simple string paths to Hugging Face models
            cifar10: tanganke/clip-vit-base-patch32_cifar10
            sun397: tanganke/clip-vit-base-patch32_sun397
            stanford-cars: tanganke/clip-vit-base-patch32_stanford-cars

            # Complex configuration with additional parameters
            custom_model:
                _target_: transformers.CLIPVisionModel.from_pretrained
                pretrained_model_name_or_path: openai/clip-vit-base-patch32
                torch_dtype: float16
        ```

        Args:
            model_name_or_config (Union[str, DictConfig]): Either a model name from the pool
                or a configuration dictionary for instantiating the model.
            *args: Additional positional arguments passed to model loading/instantiation.
            **kwargs: Additional keyword arguments passed to model loading/instantiation.

        Returns:
            CLIPVisionModel: The loaded CLIPVisionModel instance.
        """
        # Check if we have a string model name that exists in our model pool
        if (
            isinstance(model_name_or_config, str)
            and model_name_or_config in self._models
        ):
            model_name = model_name_or_config

            # handle different model configuration types
            match self._models[model_name_or_config]:
                case str() as model_path:
                    # Handle string model paths (e.g., Hugging Face model IDs)
                    if rank_zero_only.rank == 0:
                        log.info(
                            f"Loading model `{model_name}` of type `transformers.CLIPVisionModel` from {model_path}"
                        )
                    # Resolve the repository path (supports both HuggingFace and ModelScope)
                    repo_path = resolve_repo_path(
                        model_path, repo_type="model", platform=self._platform
                    )
                    # Load and return the CLIPVisionModel from the resolved path
                    return CLIPVisionModel.from_pretrained(repo_path, *args, **kwargs)

                case nn.Module() as model:
                    # Handle pre-instantiated model objects
                    if rank_zero_only.rank == 0:
                        log.info(
                            f"Returning existing model `{model_name}` of type {type(model)}"
                        )
                    return model

                case _:
                    # Handle other configuration types (e.g., DictConfig) via parent class
                    # This fallback prevents returning None when the model config doesn't
                    # match the expected string or nn.Module patterns
                    return super().load_model(model_name_or_config, *args, **kwargs)

        # If model_name_or_config is not a string in our pool, delegate to parent class
        # This handles cases where model_name_or_config is a DictConfig directly
        return super().load_model(model_name_or_config, *args, **kwargs)

    def load_train_dataset(self, dataset_name: str, *args, **kwargs):
        dataset_config = self._train_datasets[dataset_name]
        if isinstance(dataset_config, str):
            if rank_zero_only.rank == 0:
                log.info(
                    f"Loading train dataset using `datasets.load_dataset`: {dataset_config}"
                )
            dataset = self._load_dataset(dataset_config, split="train")
        else:
            dataset = super().load_train_dataset(dataset_name, *args, **kwargs)
        return dataset

    def load_val_dataset(self, dataset_name: str, *args, **kwargs):
        dataset_config = self._val_datasets[dataset_name]
        if isinstance(dataset_config, str):
            if rank_zero_only.rank == 0:
                log.info(
                    f"Loading validation dataset using `datasets.load_dataset`: {dataset_config}"
                )
            dataset = self._load_dataset(dataset_config, split="validation")
        else:
            dataset = super().load_val_dataset(dataset_name, *args, **kwargs)
        return dataset

    def load_test_dataset(self, dataset_name: str, *args, **kwargs):
        dataset_config = self._test_datasets[dataset_name]
        if isinstance(dataset_config, str):
            if rank_zero_only.rank == 0:
                log.info(
                    f"Loading test dataset using `datasets.load_dataset`: {dataset_config}"
                )
            dataset = self._load_dataset(dataset_config, split="test")
        else:
            dataset = super().load_test_dataset(dataset_name, *args, **kwargs)
        return dataset

    def _load_dataset(self, name: str, split: str):
        """
        Load a dataset by its name and split.

        Args:
            dataset_name (str): The name of the dataset.
            split (str): The split of the dataset to load (e.g., "train", "validation", "test").

        Returns:
            Dataset: The loaded dataset.
        """
        datset_dir = resolve_repo_path(
            name, repo_type="dataset", platform=self._platform
        )
        dataset = load_dataset(datset_dir, split=split)
        return dataset
