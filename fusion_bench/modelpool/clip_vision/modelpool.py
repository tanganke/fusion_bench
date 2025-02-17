import logging
from copy import deepcopy
from typing import Optional, Union

from datasets import load_dataset
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from typing_extensions import override

from fusion_bench.utils import instantiate, timeit_context

from ..base_pool import BaseModelPool

log = logging.getLogger(__name__)


class CLIPVisionModelPool(BaseModelPool):
    """
    A model pool for managing Hugging Face's CLIP Vision models.

    This class extends the base `ModelPool` class and overrides its methods to handle
    the specifics of the CLIP Vision models provided by the Hugging Face Transformers library.
    """

    _config_mapping = BaseModelPool._config_mapping | {"_processor": "processor"}

    def __init__(
        self,
        models: DictConfig,
        *,
        processor: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)

        self._processor = processor

    def load_processor(self, *args, **kwargs) -> CLIPProcessor:
        assert self._processor is not None, "Processor is not defined in the config"
        if isinstance(self._processor, str):
            log.info(f"Loading `transformers.CLIPProcessor`: {self._processor}")
            processor = CLIPProcessor.from_pretrained(self._processor)
        else:
            processor = instantiate(self._processor, *args, **kwargs)
        return processor

    def load_clip_model(self, model_name: str, *args, **kwargs) -> CLIPModel:
        model_config = self._models[model_name]

        if isinstance(model_config, str):
            log.info(f"Loading `transformers.CLIPModel`: {model_config}")
            clip_model = CLIPModel.from_pretrained(model_config, *args, **kwargs)
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
        This method is used to load a CLIPVisionModel from the model pool.

        Example configuration could be:

        ```yaml
        models:
            cifar10: tanganke/clip-vit-base-patch32_cifar10
            sun397: tanganke/clip-vit-base-patch32_sun397
            stanford-cars: tanganke/clip-vit-base-patch32_stanford-cars
        ```

        Args:
            model_name_or_config (Union[str, DictConfig]): The name of the model or the model configuration.

        Returns:
            CLIPVisionModel: The loaded CLIPVisionModel.
        """
        if (
            isinstance(model_name_or_config, str)
            and model_name_or_config in self._models
        ):
            model = self._models[model_name_or_config]
            if isinstance(model, str):
                log.info(f"Loading `transformers.CLIPVisionModel`: {model}")
                return CLIPVisionModel.from_pretrained(model, *args, **kwargs)
            if isinstance(model, nn.Module):
                log.info(f"Returning existing model: {model}")
                return model

        # If the model is not a string, we use the default load_model method
        return super().load_model(model_name_or_config, *args, **kwargs)

    def load_train_dataset(self, dataset_name: str, *args, **kwargs):
        dataset_config = self._train_datasets[dataset_name]
        if isinstance(dataset_config, str):
            log.info(
                f"Loading train dataset using `datasets.load_dataset`: {dataset_config}"
            )
            dataset = load_dataset(dataset_config, split="train")
        else:
            dataset = super().load_train_dataset(dataset_name, *args, **kwargs)
        return dataset

    def load_val_dataset(self, dataset_name: str, *args, **kwargs):
        dataset_config = self._val_datasets[dataset_name]
        if isinstance(dataset_config, str):
            log.info(
                f"Loading validation dataset using `datasets.load_dataset`: {dataset_config}"
            )
            dataset = load_dataset(dataset_config, split="validation")
        else:
            dataset = super().load_val_dataset(dataset_name, *args, **kwargs)
        return dataset

    def load_test_dataset(self, dataset_name: str, *args, **kwargs):
        dataset_config = self._test_datasets[dataset_name]
        if isinstance(dataset_config, str):
            log.info(
                f"Loading test dataset using `datasets.load_dataset`: {dataset_config}"
            )
            dataset = load_dataset(dataset_config, split="test")
        else:
            dataset = super().load_test_dataset(dataset_name, *args, **kwargs)
        return dataset
