import functools
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast  # noqa: F401

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils.data import InfiniteDataLoader

log = logging.getLogger(__name__)

TensorOrModule = TypeVar("TensorOrModule", torch.Tensor, torch.nn.Module, Any)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIPClassificationMixin(LightningFabricMixin):
    """
    This mixin provides methods to classify images using the CLIP model.

    Attributes need to be set by the inheriting class:

    - `_dataloader_kwargs` (Dict[str, Any]): Keyword arguments for the dataloader.
    - `modelpool` (CLIPVisionModelPool): The model pool containing the CLIP models.
    - `zeroshot_weights_cache_dir` (Optional[str]): The directory to cache the zero-shot weights.
    """

    _dataloader_kwargs: Dict[str, Any] = {}
    # the modelpool is set by inheriting class
    modelpool: CLIPVisionModelPool = None
    _clip_processor: CLIPProcessor = None
    # a dict of zeroshot weights for each task, each key is the task name
    zeroshot_weights_cache_dir: str = "outputs/cache/clip_zeroshot_weights"
    zeroshot_weights: Dict[str, torch.Tensor] = {}

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            assert self.modelpool is not None, "Model pool is not set"
            self._clip_processor = self.modelpool.load_processor()
        return self._clip_processor

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        loader = DataLoader(
            CLIPDataset(self.modelpool.load_test_dataset(task), self.clip_processor),
            **self._dataloader_kwargs,
        )
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    @torch.no_grad()
    def setup_zero_shot_classification_head(
        self,
        clip_processor: Optional[CLIPProcessor] = None,
        clip_model: Optional[CLIPModel] = None,
        task_names: Optional[List[str]] = None,
    ):
        if clip_model is None:
            if self.modelpool.has_pretrained:
                clip_model = self.modelpool.load_clip_model("_pretrained_")
            else:
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

        # get cache directory
        if self.modelpool.has_pretrained:
            model_name = self.modelpool.get_model_config(
                "_pretrained_"
            ).pretrained_model_name_or_path
        else:
            model_name = self.modelpool.get_model_config(
                self.modelpool.model_names[0]
            ).pretrained_model_name_or_path
        cache_dir = os.path.join(
            self.zeroshot_weights_cache_dir,
            os.path.normpath(model_name.split("/")[-1]),
        )
        if not os.path.exists(cache_dir):
            log.info(
                f"Creating cache directory for zero-shot classification head at {cache_dir}"
            )
            os.makedirs(cache_dir)

        log.info(f"cache directory for zero-shot classification head: {cache_dir}")
        for task in tqdm(
            self.modelpool.model_names if task_names is None else task_names,
            "Setting up zero-shot classification head",
            disable=not self.fabric.is_global_zero,
        ):
            zeroshot_weights = None
            if self.fabric.is_global_zero:
                cache_file = os.path.join(
                    cache_dir, os.path.normpath(f"{task}_zeroshot_weights.pt")
                )
                if os.path.exists(cache_file):
                    log.info(f"Loading cached zeroshot weights for task: {task}")
                    zeroshot_weights = torch.load(
                        cache_file,
                        map_location="cpu",
                        weights_only=True,
                    ).detach()
                else:
                    log.info(
                        f"Construct zero shot classification head for task: {task}"
                    )
                    classnames, templates = get_classnames_and_templates(task)
                    clip_classifier.set_classification_task(classnames, templates)
                    zeroshot_weights = clip_classifier.zeroshot_weights.detach().clone()
                    log.info(f"save zeroshot weights to {cache_file}")
                    torch.save(zeroshot_weights, cache_file)

            self.fabric.barrier()
            self.zeroshot_weights[task] = self.fabric.broadcast(zeroshot_weights, src=0)
            self.zeroshot_weights[task] = self.to_device(self.zeroshot_weights[task])

        del clip_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_logits(
        self,
        module: Union[nn.Module, CLIPVisionModel],
        images: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image
