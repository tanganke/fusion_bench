from typing import Callable, Iterable, List

import torch
from torch import Tensor, nn
from torch.types import _device
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel, CLIPTextModel


default_templates = [
    lambda c: f"a photo of a {c}",
]


class HFCLIPClassifier(nn.Module):
    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
    ):
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

    @property
    def text_model(self):
        return self.clip_model.text_model

    @property
    def vision_model(self):
        return self.clip_model.vision_model

    def set_classification_task(
        self,
        classnames: List[str],
        templates: List[Callable[[str], str]] = default_templates,
    ):
        processor = self.processor

        self.classnames = classnames
        self.templates = templates

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                text = [template(classname) for template in templates]
                inputs = processor(text=text, return_tensors="pt", padding=True)

                embeddings = self.text_model(**inputs)[1]
                embeddings = self.clip_model.text_projection(embeddings)

                # normalize embeddings
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.mean(dim=0)
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0)

        self.zeroshot_weights = zeroshot_weights

    def forward(self, images):
        if self.zeroshot_weights is None:
            raise ValueError("Must set classification task before forward pass")
        text_embeds = self.zeroshot_weights

        image_embeds = self.vision_model(images)[1]
        image_embeds = self.clip_model.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image
