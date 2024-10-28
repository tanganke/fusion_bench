import logging
from typing import Dict, List, cast  # noqa: F401

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import CLIPProcessor

from fusion_bench.dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin

from .fisher_merging import FisherMergingAlgorithm, get_param_squared_gradients

log = logging.getLogger(__name__)


class FisherMergingForCLIPVisionModel(
    CLIPClassificationMixin,
    FisherMergingAlgorithm,
):
    """
    Implements Fisher Merging for CLIP Vision Models.

    This class extends the FisherMergingAlgorithm and CLIPClassificationMixin to handle
    the specifics of merging CLIP Vision models using Fisher weights.
    """

    _clip_processor: CLIPProcessor = None
    zeroshot_weights = {}

    _config_mapping = FisherMergingAlgorithm._config_mapping | {
        "zeroshot_weights_cache_dir": "zeroshot_weights_cache_dir",
        "_dataloader_kwargs": "dataloader_kwargs",
    }

    def __init__(
        self,
        *,
        exclude_param_names_regex,
        normalize_fisher_weight,
        minimal_fisher_weight,
        num_fisher_examples,
        dataloader_kwargs: DictConfig,
        zeroshot_weights_cache_dir=None,
        **kwargs,
    ):
        """
        Initialize the FisherMergingForCLIPVisionModel with the given configuration.

        Args:
            exclude_param_names_regex (list): List of regex patterns to exclude certain parameter names.
            normalize_fisher_weight (bool): Whether to normalize Fisher weights.
            minimal_fisher_weight (float): Minimal value for Fisher weights to avoid numerical issues.
            num_fisher_examples (int): Number of examples to compute Fisher weights.
            dataloader_kwargs (DictConfig): Configuration for the dataloader.
            zeroshot_weights_cache_dir (str, optional): Directory to cache zero-shot weights. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            exclude_param_names_regex=exclude_param_names_regex,
            normalize_fisher_weight=normalize_fisher_weight,
            minimal_fisher_weight=minimal_fisher_weight,
            num_fisher_examples=num_fisher_examples,
        )
        self._dataloader_kwargs = dataloader_kwargs
        self.zeroshot_weights_cache_dir = zeroshot_weights_cache_dir
        for key, value in kwargs.items():
            log.warning(f"Unused argument: {key}={value}")
            setattr(self, key, value)

    def on_fisher_merging_start(self):
        """
        Setup the zero-shot classification head before starting the Fisher merging process.
        """
        self.setup_zero_shot_classification_head()

    def compute_logits(self, module, batch, task: str) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module (Module): The model module.
            batch (tuple): A batch of data containing images and labels.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        images, _ = batch
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

    def get_fisher_weights(
        self,
        model_name: str,
        model: Module,
        train_dataset,
        param_names_to_merge: List[str],
    ) -> Dict[str, Tensor]:
        """
        Compute the Fisher weights for the given model and training dataset.

        Args:
            model_name (str): The name of the model.
            model (Module): The model module.
            train_dataset: The training dataset.
            param_names_to_merge (List[str]): List of parameter names to merge.

        Returns:
            Dict[str, Tensor]: The computed Fisher weights for each parameter.
        """
        # setup dataloader
        train_dataset = CLIPDataset(train_dataset, self.clip_processor)
        train_dataloader = DataLoader(train_dataset, **self._dataloader_kwargs)
        if self.fabric is not None:
            train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
            model = self.fabric.setup(model)
        num_fisher_examples = self.config.num_fisher_examples
        if num_fisher_examples % train_dataloader.batch_size != 0:
            print(
                "warning: the number of examples for computing fisher cannot be fully divided by the batch size for model, "
                "which may lead to a slightly different number of the actually used examples."
            )
        num_computed_examples = 0
        batches_fisher_weights_list = []
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc="computing fisher weights",
            total=num_fisher_examples // train_dataloader.batch_size,
        ):
            if num_computed_examples >= num_fisher_examples:
                break
            logits = self.compute_logits(model, batch, model_name)
            # Tensor, shape (batch_size, num_label_classes)

            # compute fisher weights for classification task
            # use detach() to detach from the computation graph
            # Tensor, shape (batch_size, num_label_classes)
            labels_probabilities = torch.softmax(logits, dim=-1).detach()
            labels_log_probabilities = torch.log_softmax(logits, dim=-1)
            # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
            labels_expectations = (
                torch.sqrt(labels_probabilities) * labels_log_probabilities
            )
            # sum over label classes and batch dimension
            sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
            model.zero_grad()
            sum_labels_expectations.backward()
            # dict, fisher weights of a batch
            batch_fisher_weights = get_param_squared_gradients(
                model=model, param_names_to_merge=param_names_to_merge
            )

            # move fisher weights to cpu to save GPU memory
            for key, weights in batch_fisher_weights.items():
                batch_fisher_weights[key] = weights.detach().cpu()

            batches_fisher_weights_list.append(batch_fisher_weights)
            num_computed_examples += batch[0].size(0)

        model_to_merge_fisher_weights = {}
        for batch_fisher_weights in batches_fisher_weights_list:
            for key in batch_fisher_weights:
                if key not in model_to_merge_fisher_weights:
                    model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                else:
                    model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

        # mean over batches
        for key in model_to_merge_fisher_weights:
            model_to_merge_fisher_weights[key] /= num_computed_examples
            model_to_merge_fisher_weights[key] = (
                model_to_merge_fisher_weights[key].detach().cpu()
            )
        return model_to_merge_fisher_weights
