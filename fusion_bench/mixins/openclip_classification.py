import functools
import logging
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Literal, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool import OpenCLIPVisionModelPool
from fusion_bench.models.open_clip import (
    ClassificationHead,
    ImageClassifier,
    ImageEncoder,
)
from fusion_bench.utils.data import InfiniteDataLoader

log = logging.getLogger(__name__)


class OpenCLIPClassificationMixin(LightningFabricMixin):

    _train_processor = None
    _test_processor = None
    dataloader_kwargs: DictConfig
    modelpool: OpenCLIPVisionModelPool
    zero_shot_heads: Dict[str, ClassificationHead] = {}

    def _init_processor(self, encoder: Optional["ImageEncoder"] = None):
        """
        Initialize the CLIP processors for training and testing.
        """
        if encoder is None:
            encoder: "ImageEncoder" = self.modelpool.load_pretrained_or_first_model()
        self._train_processor = encoder.train_preprocess
        self._test_processor = encoder.val_preprocess
        return self._train_processor, self._test_processor

    def get_clip_processor(self, stage: Literal["train", "test"]):
        """
        Get the CLIP processor, loading it from the model pool if necessary.

        Returns:
            CLIPProcessor: The CLIP processor for image and text preprocessing.

        Raises:
            AssertionError: If the model pool is not set.
        """
        if stage == "train":
            if self._train_processor is None:
                self._init_processor()
            return self._train_processor
        elif stage == "test":
            if self._test_processor is None:
                self._init_processor()
            return self._test_processor
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def setup_zero_shot_classification_head(
        self,
        task_names: Optional[List[str]] = None,
        freeze: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        # check task names consistency across processes
        _task_names = self.fabric.broadcast(task_names, src=0)
        if not self.fabric.is_global_zero and task_names != _task_names:
            raise ValueError("The `task_names` must be the same across all processes.")

        for task in tqdm(
            self.modelpool.model_names if task_names is None else task_names,
            "Setting up zero-shot classification head",
            disable=not self.fabric.is_global_zero,
        ):
            head = self.modelpool.load_classification_head(task)
            if freeze:
                head.requires_grad_(False)
            if dtype is not None:
                head = head.to(dtype=dtype)
            self.zero_shot_heads[task] = self.to_device(head)

    def set_clip_processor(self, stage: Literal["train", "test"], processor: Callable):
        """
        Set the CLIP processor for a specific stage.

        Args:
            stage (Literal["train", "test"]): The stage for which to set the processor.
            processor (Callable): The CLIP processor to set.
        """
        if stage == "train":
            self._train_processor = processor
        elif stage == "test":
            self._test_processor = processor
        else:
            raise ValueError(f"Invalid stage: {stage}")

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
            self.modelpool.load_test_dataset(task),
            processor=self.get_clip_processor(stage="test"),
        )
        # create the dataloader
        loader = DataLoader(clip_dataset, **dataloader_kwargs)
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def compute_logits(
        self,
        module: ImageClassifier,
        images,
        task: str,
    ):
        """
        Compute the logits for a batch of images using the provided module and task.

        Args:
            module (ImageClassifier): The image classification module to use for computing logits.
            images (torch.Tensor): The batch of images for which to compute logits.
            task (str): The task identifier to specify which classification head to use.

        Returns:
            torch.Tensor: The computed logits for the input images.
        """
        if len(self.zero_shot_heads) == 0:
            self.setup_zero_shot_classification_head()
        task_head = self.zero_shot_heads[task]
        features = module(images)
        logits = task_head(features)
        return logits
