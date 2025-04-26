import logging
from typing import TYPE_CHECKING, Callable, Iterable, List  # noqa: F401

import torch
from torch import Tensor, nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import BaseModelOutputWithPooling

from fusion_bench.utils.devices import get_device

if TYPE_CHECKING:
    from fusion_bench.models.surgery.surgerymodelwrapper import SurgeryModelWrapper

log = logging.getLogger(__name__)

default_templates = [
    lambda c: f"a photo of a {c}",
]


class HFCLIPClassifier(nn.Module):
    """
    A classifier based on the CLIP (Contrastive Language-Image Pre-training) model.

    This class wraps a CLIP model and provides functionality for image classification
    using zero-shot learning. It allows setting a classification task with custom
    class names and text templates.

    Attributes:
        clip_model (CLIPModel): The underlying CLIP model.
        processor (CLIPProcessor): The CLIP processor for preparing inputs.
        zeroshot_weights (Tensor): Computed text embeddings for zero-shot classification.
        classnames (List[str]): List of class names for the current classification task.
        templates (List[Callable[[str], str]]): List of template functions for generating text prompts.

    """

    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        extra_module=None,
    ):
        """
        Initialize the HFCLIPClassifier.

        Args:
            clip_model (CLIPModel): The CLIP model to use for classification.
            processor (CLIPProcessor): The CLIP processor for preparing inputs.
        """
        super().__init__()
        # we only fine-tune the vision model
        clip_model.visual_projection.requires_grad_(False)
        clip_model.text_model.requires_grad_(False)
        clip_model.text_projection.requires_grad_(False)
        clip_model.logit_scale.requires_grad_(False)

        self.clip_model = clip_model
        self.processor = processor
        self.register_buffer(
            "zeroshot_weights",
            None,
            persistent=False,
        )

        self.extra_module = extra_module

    @property
    def text_model(self):
        """Get the text model component of CLIP."""
        return self.clip_model.text_model

    @property
    def vision_model(self):
        """Get the vision model component of CLIP."""
        return self.clip_model.vision_model

    def set_classification_task(
        self,
        classnames: List[str],
        templates: List[Callable[[str], str]] = default_templates,
    ):
        """
        Set up the zero-shot classification task.

        This method computes text embeddings for the given class names using the
        provided templates. These embeddings are then used for classification.

        Args:
            classnames (List[str]): List of class names for the classification task.
            templates (List[Callable[[str], str]], optional): List of template functions
                for generating text prompts. Defaults to `default_templates`, i.e.
                ["a photo of a {classname}"].
        """
        processor = self.processor

        self.classnames = classnames
        self.templates = templates

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                text = [template(classname) for template in templates]
                inputs = processor(text=text, return_tensors="pt", padding=True)
                inputs = {
                    k: v.to(get_device(self.text_model)) for k, v in inputs.items()
                }
                embeddings = self.text_model(**inputs)[1]
                embeddings = self.clip_model.text_projection(embeddings)

                # normalize embeddings
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.mean(dim=0)
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0)

        self.zeroshot_weights = zeroshot_weights

    def forward(
        self,
        images: Tensor,
        return_image_embeds=False,
        return_dict=False,
        task_name=None,
    ):
        """
        Perform forward pass for zero-shot image classification.

        This method computes image embeddings for the input images and calculates
        the similarity with the pre-computed text embeddings to produce classification logits.

        Args:
            images (Tensor): Input images to classify.
            return_image_embeds (bool): Whether to return the image embeddings.
            return_dict (bool): Whether to return a dictionary with logits and image embeddings.
            task_name (Optional[str]): The name of the task.

        Returns:
            Tensor: Classification logits for each input image.

        Raises:
            ValueError: If the classification task hasn't been set using set_classification_task.
        """
        if self.zeroshot_weights is None:
            raise ValueError("Must set classification task before forward pass")
        text_embeds = self.zeroshot_weights

        image_embeds = self.get_image_features(images)
        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        if (
            hasattr(self.vision_model, "is_surgery_model")
            and self.vision_model.is_surgery_model
        ):
            # Dealing with the surgery model, for more details, please refer to:
            # (ICML 2024) Yang, et.al. Representation Surgery for Multi-Task Model Merging
            # https://arxiv.org/abs/2402.02705
            self.vision_model: "SurgeryModelWrapper" = self.vision_model
            image_embeds, _, _ = self.vision_model.compute_surgery_features(
                image_embeds, dataset_name=task_name
            )

        # cosine similarity
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        if return_dict:
            ret = {"logits": logits_per_image}
            if return_image_embeds:
                ret.update({"image_embeds": image_embeds})
            return ret
        else:
            if return_image_embeds:
                return logits_per_image, image_embeds
            else:
                return logits_per_image

    def get_image_features(self, images: Tensor) -> Tensor:
        """
        Compute the image embeddings.

        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        """

        image_embeds = self.vision_model(images)
        if isinstance(image_embeds, Tensor):
            pass
        elif isinstance(image_embeds, BaseModelOutputWithPooling):
            image_embeds = image_embeds[1]
        image_embeds = self.clip_model.visual_projection(image_embeds)
        return image_embeds
