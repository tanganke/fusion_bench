import logging
import pickle
import sys
from typing import Callable, Optional, Union, cast

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.open_clip import ClassificationHead, ImageEncoder
from fusion_bench.utils import instantiate
from fusion_bench.utils.expr import is_expr_match
from fusion_bench.utils.packages import _get_package_version, compare_versions

log = logging.getLogger(__name__)

# Add flag to track if warning has been shown
_openclip_version_warning_shown = False


def _check_and_redirect_open_clip_modeling():
    global _openclip_version_warning_shown
    if compare_versions(_get_package_version("open-clip-torch").__str__(), "2.0.2") > 0:
        if not _openclip_version_warning_shown:
            log.warning(
                "OpenCLIP version is greater than 2.0.2. This may cause issues with the modelpool."
            )
            _openclip_version_warning_shown = True
        import open_clip.model
        import open_clip.transformer

        if not hasattr(open_clip.model, "VisualTransformer"):
            open_clip.model.VisualTransformer = open_clip.model.VisionTransformer
        if not hasattr(open_clip.model, "Transformer"):
            open_clip.model.Transformer = open_clip.transformer.Transformer
        if not hasattr(open_clip.model, "ResidualAttentionBlock"):
            open_clip.model.ResidualAttentionBlock = (
                open_clip.transformer.ResidualAttentionBlock
            )

    try:
        import src
        import src.modeling
    except ImportError:
        if "src" not in sys.modules:
            # redirect the import of `src` to `fusion_bench.models.open_clip`
            import fusion_bench.models.open_clip as open_clip

            sys.modules["src"] = open_clip
            log.warning(
                "`src` is not imported."
                "Redirecting the import to `fusion_bench.models.open_clip`"
            )
        if "src.modeling" not in sys.modules:
            # redirect the import of `src.modeling` to `fusion_bench.models.open_clip.modeling`
            import fusion_bench.models.open_clip.modeling as open_clip_modeling

            sys.modules["src.modeling"] = open_clip_modeling
            log.warning(
                "`src.modeling` is not imported."
                "Redirecting the import to `fusion_bench.models.open_clip.modeling`"
            )


def load_classifier_head(model_config: Union[str, DictConfig], *args, **kwargs):
    if isinstance(model_config, str):
        _check_and_redirect_open_clip_modeling()
        log.info(f"Loading `ClassificationHead` from {model_config}")
        weights_only = kwargs["weights_only"] if "weights_only" in kwargs else False
        head = torch.load(model_config, weights_only=weights_only, *args, **kwargs)
    elif isinstance(model_config, nn.Module):
        log.info(f"Returning existing model: {model_config}")
        head = model_config
    else:
        head = instantiate(model_config, *args, **kwargs)
    head = cast(ClassificationHead, head)
    return head


class OpenCLIPVisionModelPool(BaseModelPool):
    """
    A model pool for managing OpenCLIP Vision models (models from task vector paper).
    """

    _train_processor = None
    _test_processor = None

    def __init__(
        self,
        models: DictConfig,
        classification_heads: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(models, **kwargs)
        self._classification_heads = classification_heads

    @property
    def train_processor(self):
        if self._train_processor is None:
            encoder: ImageEncoder = self.load_pretrained_or_first_model()
            self._train_processor = encoder.train_preprocess
            if self._test_processor is None:
                self._test_processor = encoder.val_preprocess
        return self._train_processor

    @property
    def test_processor(self):
        if self._test_processor is None:
            encoder: ImageEncoder = self.load_pretrained_or_first_model()
            if self._train_processor is None:
                self._train_processor = encoder.train_preprocess
            self._test_processor = encoder.val_preprocess
        return self._test_processor

    def load_model(
        self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
    ) -> ImageEncoder:
        R"""
        The model config can be:

        - A string, which is the path to the model checkpoint in pickle format. Load directly using `torch.load`.
        - {"model_name": str, "pickle_path": str}, load the model from the binary file (pickle format). This will first construct the model using `ImageEncoder(model_name)`, and then load the state dict from model located in the pickle file.
        - {"model_name": str, "state_dict_path": str}, load the model from the state dict file. This will first construct the model using `ImageEncoder(model_name)`, and then load the state dict from the file.
        - Default, load the model using `instantiate` from hydra.
        """
        if (
            isinstance(model_name_or_config, str)
            and model_name_or_config in self._models
        ):
            model_config = self._models[model_name_or_config]
        else:
            model_config = model_name_or_config
        if isinstance(model_config, DictConfig):
            model_config = OmegaConf.to_container(model_config, resolve=True)

        if isinstance(model_config, str):
            # the model config is a string, which is the path to the model checkpoint in pickle format
            # load the model using `torch.load`
            # this is the original usage in the task arithmetic codebase
            _check_and_redirect_open_clip_modeling()
            log.info(f"loading ImageEncoder from {model_config}")
            weights_only = kwargs["weights_only"] if "weights_only" in kwargs else False
            try:
                encoder = torch.load(
                    model_config, weights_only=weights_only, *args, **kwargs
                )
            except RuntimeError as e:
                encoder = pickle.load(open(model_config, "rb"))
        elif is_expr_match({"model_name": str, "pickle_path": str}, model_config):
            # the model config is a dictionary with the following keys:
            # - model_name: str, the name of the model
            # - pickle_path: str, the path to the binary file (pickle format)
            # load the model from the binary file (pickle format)
            # this is useful when you use a newer version of torchvision
            _check_and_redirect_open_clip_modeling()
            log.info(
                f"loading ImageEncoder of {model_config['model_name']} from {model_config['pickle_path']}"
            )
            weights_only = kwargs["weights_only"] if "weights_only" in kwargs else False
            try:
                encoder = torch.load(
                    model_config["pickle_path"],
                    weights_only=weights_only,
                    *args,
                    **kwargs,
                )
            except RuntimeError as e:
                encoder = pickle.load(open(model_config["pickle_path"], "rb"))
            _encoder = ImageEncoder(model_config["model_name"])
            _encoder.load_state_dict(encoder.state_dict())
            encoder = _encoder
        elif is_expr_match({"model_name": str, "state_dict_path": str}, model_config):
            # the model config is a dictionary with the following keys:
            # - model_name: str, the name of the model
            # - state_dict_path: str, the path to the state dict file
            # load the model from the state dict file
            log.info(
                f"loading ImageEncoder of {model_config['model_name']} from {model_config['state_dict_path']}"
            )
            encoder = ImageEncoder(model_config["model_name"])
            encoder.load_state_dict(
                torch.load(
                    model_config["state_dict_path"], weights_only=True, *args, **kwargs
                )
            )
        elif isinstance(model_config, nn.Module):
            # the model config is an existing model
            log.info(f"Returning existing model: {model_config}")
            encoder = model_config
        else:
            encoder = super().load_model(model_name_or_config, *args, **kwargs)
        encoder = cast(ImageEncoder, encoder)

        # setup the train and test processors
        if self._train_processor is None and hasattr(encoder, "train_preprocess"):
            self._train_processor = encoder.train_preprocess
        if self._test_processor is None and hasattr(encoder, "val_preprocess"):
            self._test_processor = encoder.val_preprocess

        return encoder

    def load_classification_head(
        self, model_name_or_config: Union[str, DictConfig], *args, **kwargs
    ) -> ClassificationHead:
        R"""
        The model config can be:

        - A string, which is the path to the model checkpoint in pickle format. Load directly using `torch.load`.
        - Default, load the model using `instantiate` from hydra.
        """
        if (
            isinstance(model_name_or_config, str)
            and model_name_or_config in self._classification_heads
        ):
            model_config = self._classification_heads[model_name_or_config]
        else:
            model_config = model_name_or_config

        head = load_classifier_head(model_config, *args, **kwargs)
        return head

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
