import os
from typing import List, cast

import torch
from transformers import AutoModel, CLIPModel, CLIPProcessor
from trl.import_utils import is_npu_available, is_xpu_available


class PickScoreScorer(torch.nn.Module):
    """
    References:
        - Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation.
            http://arxiv.org/abs/2305.01569
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        model_id: str = "yuvalkirstain/PickScore_v1",
        processor_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ):
        super().__init__()
        self.dtype = dtype

        self.processor = CLIPProcessor.from_pretrained(processor_name_or_path)
        self.model = (
            cast(CLIPModel, AutoModel.from_pretrained(model_id))
            .eval()
            .to(dtype=self.dtype)
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: List[str]):
        """
        Scores the given images based on their relevance to the given prompts.

        Args:
            images (torch.Tensor): The images to score.
            prompts (List[str]): The prompts to score the images against.

        Returns:
            scores (torch.Tensor): The scores of the images.
        """
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return scores


def pickscore_scorer(
    dtype: torch.dtype = torch.float32,
    hub_model_id: str = "yuvalkirstain/PickScore_v1",
):
    """
    Creates a scoring function that scores images based on their relevance to a set of prompts.

    Args:
        dtype (torch.dtype, optional): The data type to use for the computations. Defaults to torch.float32.
        hub_model_id (str, optional): The id of the pretrained model to use. Defaults to "yuvalkirstain/PickScore_v1".

    Returns:
        _fn (function): The scoring function.
    """
    scorer = PickScoreScorer(
        dtype=dtype,
        model_id=hub_model_id,
    )
    if is_npu_available():
        scorer = scorer.npu()
    elif is_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images: torch.Tensor, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores: torch.Tensor = scorer(images, prompts)
        return scores, {}

    return _fn
