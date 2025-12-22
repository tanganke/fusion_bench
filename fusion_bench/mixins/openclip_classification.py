import functools
import logging
from typing import Dict, Iterator, Optional

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.models.open_clip import ImageClassifier, ImageEncoder
from fusion_bench.utils.data import InfiniteDataLoader

log = logging.getLogger(__name__)


class OpenCLIPClassificationMixin(LightningFabricMixin):
    _train_processor = None
    _test_processor = None
    _clip_processor = None
    dataloader_kwargs: DictConfig

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
