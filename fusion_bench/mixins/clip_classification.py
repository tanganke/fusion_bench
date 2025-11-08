import functools
import logging
import os
from copy import deepcopy
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench import cache_with_joblib
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils.data import InfiniteDataLoader

if TYPE_CHECKING:
    from transformers.models.clip.modeling_clip import CLIPVisionTransformer

log = logging.getLogger(__name__)

# disable tokenizers parallelism by default to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIPClassificationMixin(LightningFabricMixin):
    """
    This mixin provides methods to classify images using the CLIP model.

    Attributes need to be set by the inheriting class:

    - `_dataloader_kwargs` (Dict[str, Any]): Keyword arguments for the dataloader.
    - `modelpool` (CLIPVisionModelPool): The model pool containing the CLIP models.
    """

    dataloader_kwargs: Dict[str, Any] = {}
    # the modelpool is set by inheriting class
    modelpool: CLIPVisionModelPool = None
    _clip_processor: CLIPProcessor = None
    # a dict of zeroshot weights for each task, each key is the task name
    zeroshot_weights: Dict[str, torch.Tensor] = {}
    whether_setup_zero_shot_classification_head = False

    @property
    def clip_processor(self):
        """
        Get the CLIP processor, loading it from the model pool if necessary.

        Returns:
            CLIPProcessor: The CLIP processor for image and text preprocessing.

        Raises:
            AssertionError: If the model pool is not set.
        """
        if self._clip_processor is None:
            assert self.modelpool is not None, "Model pool is not set"
            self._clip_processor = self.modelpool.load_processor()
        return self._clip_processor

    @functools.cache
    def get_shuffled_test_loader_iter(
        self,
        task: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **loader_kwargs,
    ) -> Iterator:
        """
        Get an iterator for a shuffled test DataLoader.

        This method creates a DataLoader for the test dataset of the specified task,
        with shuffling enabled. It allows for optional customization of batch size,
        number of workers, and other DataLoader keyword arguments.

        Args:
            task (str): The task identifier for which the test dataset is to be loaded.
            batch_size (Optional[int]): The batch size to use for the DataLoader. If None, the default batch size is used.
            num_workers (Optional[int]): The number of worker processes to use for data loading. If None, the default number of workers is used.
            **loader_kwargs: Additional keyword arguments to pass to the DataLoader.

        Returns:
            Iterator: An iterator over the shuffled test DataLoader.
        """
        # get dataloader kwargs
        dataloader_kwargs = self.dataloader_kwargs.copy()
        dataloader_kwargs["shuffle"] = True
        if batch_size is not None:
            dataloader_kwargs["batch_size"] = batch_size
        if num_workers is not None:
            dataloader_kwargs["num_workers"] = num_workers
        dataloader_kwargs.update(loader_kwargs)

        # get the test dataset
        clip_dataset = CLIPDataset(
            self.modelpool.load_test_dataset(task), self.clip_processor
        )
        # create the dataloader
        loader = DataLoader(clip_dataset, **dataloader_kwargs)
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    @torch.no_grad()
    def setup_zero_shot_classification_head(
        self,
        clip_processor: Optional[CLIPProcessor] = None,
        clip_model: Optional[CLIPModel] = None,
        task_names: Optional[List[str]] = None,
    ):
        """
        Initializes a zero-shot classification head.

        This method constructs a zero-shot classification head by generating text embeddings for each class name using a set of templates.
        These embeddings function as the weights of the classification layer. The method also extracts the `visual_projection` and `logit_scale`
        from the provided CLIP model, which are necessary for calculating the final logits.

        Args:
            clip_processor (Optional[CLIPProcessor]): The processor for the CLIP model. If not provided, it is loaded from the model pool.
            clip_model (Optional[CLIPModel]): The CLIP model to use. If not provided, a pretrained model is loaded from the model pool.
            task_names (Optional[List[str]]): A list of task names to set up the classification head for. If not provided, all models in the model pool will be used.
        """
        # make sure the task names are equal across all processes
        _task_names = self.fabric.broadcast(task_names, src=0)
        if not self.fabric.is_global_zero and task_names != _task_names:
            raise ValueError("The `task_names` must be the same across all processes.")

        self.whether_setup_zero_shot_classification_head = True
        # load clip model if not provided
        if clip_model is None:
            if self.modelpool.has_pretrained:
                clip_model = self.modelpool.load_clip_model("_pretrained_")
            else:
                log.warning(
                    f"No pretrained CLIP model found, using the model from the model pool: {self.modelpool.model_names[0]}."
                )
                clip_model = self.modelpool.load_clip_model(
                    self.modelpool.model_names[0]
                )
        if clip_processor is None:
            clip_processor = self.clip_processor
        clip_classifier = HFCLIPClassifier(clip_model, clip_processor)
        self.visual_projection = deepcopy(clip_model.visual_projection)
        self.visual_projection.requires_grad_(False)
        self.logit_scale_exp = clip_model.logit_scale.data.clone().exp()
        self.visual_projection = self.fabric.to_device(self.visual_projection)
        self.logit_scale_exp = self.fabric.to_device(self.logit_scale_exp)

        @cache_with_joblib()
        def construct_classification_head(task: str, model_name: str):
            log.info(
                f"Constructing zero-shot classification head for task: {task} using model: {model_name}"
            )
            nonlocal clip_classifier

            classnames, templates = get_classnames_and_templates(task)
            clip_classifier.set_classification_task(classnames, templates)
            zeroshot_weights = clip_classifier.zeroshot_weights.detach().clone()

            return zeroshot_weights

        for task in tqdm(
            self.modelpool.model_names if task_names is None else task_names,
            "Setting up zero-shot classification head",
            disable=not self.fabric.is_global_zero,
        ):
            zeroshot_weights = None
            if self.fabric.is_global_zero:
                if hasattr(clip_model, "config") and hasattr(
                    clip_model.config, "_name_or_path"
                ):
                    model_name = clip_model.config._name_or_path
                else:
                    model_name = "unknown_model"
                    log.warning(
                        "CLIP model config does not have `_name_or_path` attribute. Using 'unknown_model' as model name."
                    )
                zeroshot_weights = construct_classification_head(
                    task, model_name=model_name
                )

            self.fabric.barrier()
            self.zeroshot_weights[task] = self.fabric.broadcast(zeroshot_weights, src=0)
            self.zeroshot_weights[task] = self.to_device(self.zeroshot_weights[task])
            self.fabric.barrier()

        del clip_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_logits(
        self,
        module: Union[nn.Module, CLIPVisionModel, "CLIPVisionTransformer"],
        images: torch.Tensor,
        task: str,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the classification logits for a batch of images for a specific task.

        This method performs zero-shot classification by calculating the cosine similarity between image and text embeddings.
        The image embeddings are obtained from the provided vision model, and the text embeddings (zero-shot weights) are pre-computed for the task.
        The similarity scores are then scaled by the CLIP model's `logit_scale` to produce the final logits.

        Args:
            module (Union[nn.Module, CLIPVisionModel, "CLIPVisionTransformer"]): The vision encoder part of the CLIP model.
            images (torch.Tensor): A batch of images to classify.
            task (str): The name of the classification task.
            image_embeds (Optional[torch.Tensor]): Pre-computed image embeddings. If provided, the method skips the image encoding step.

        Returns:
            torch.Tensor: A tensor of logits for each image, with shape (batch_size, num_classes).
        """
        text_embeds = self.zeroshot_weights[task]

        if image_embeds is None:
            image_embeds = module(images)[1]
        assert isinstance(
            image_embeds, torch.Tensor
        ), f"`image_embeds` must be a tensor, but got {type(image_embeds)}"
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image

    def compute_features(
        self,
        module: Union[nn.Module, CLIPVisionModel, "CLIPVisionTransformer"],
        images: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extracts image features using CLIP's vision encoder and visual projection.

        Args:
            module (Union[nn.Module, CLIPVisionModel, "CLIPVisionTransformer"]): The CLIP vision encoder module.
            images (torch.Tensor): Input image batch to process.
            normalize (bool): Whether to normalize the image embeddings.

        Returns:
            torch.Tensor: Normalized image embeddings with dimension matching CLIP's projection space (`projection_dim` in model config).
        """
        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        if normalize:
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds
