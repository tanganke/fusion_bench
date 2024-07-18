import os
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from torch import Tensor, nn
from transformers import CLIPModel, CLIPProcessor
from trl.import_utils import is_npu_available, is_xpu_available


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed: Tensor) -> Tensor:
        """
        Forward pass through the MLP. The return value is a single scalar.
        """
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor

    Note for `model_id` and `model_filename`:

        In some implementation, the filename of the MLP model is 'sac+logos+ava1-l14-linearMSE.pth',
        which is the same as the default value of the 'model_filename' parameter in the constructor ('aesthetic-model.pth').
        It was simply renamed to 'aesthetic-model.pth' in the implementation.
        see https://huggingface.co/trl-lib/ddpo-aesthetic-predictor/commit/7f639699bec8126062148a47ecb1a4312d8e6688
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        model_id: str = "trl-lib/ddpo-aesthetic-predictor",
        model_filename: str = "aesthetic-model.pth",
    ):
        """
        Initialize the AestheticScorer class.

        Args:
            dtype (torch.dtype): The data type of the tensors.
            model_id (str, optional): The ID of the model to download. Defaults to "trl-lib/ddpo-aesthetic-predictor".
            model_filename (str, optional): The filename of the model to download. Defaults to "aesthetic-model.pth". This is the same as 'sac+logos+ava1-l14-linearMSE.pth' in some implementations.
        """
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images: Tensor) -> Tensor:
        """
        Process the given images and return their aesthetic scores.

        This method processes the images using the CLIP model, normalizes the embeddings,
        and then passes them through a MLP to get the aesthetic scores.

        Args:
            images (torch.Tensor): A batch of images to process.

        Returns:
            Tensor: The aesthetic scores of the images. Return shape is (batch_size,).
        """
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            k: cast(Tensor, v).to(self.dtype).to(device) for k, v in inputs.items()
        }
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(
    dtype: torch.dtype = torch.float32,
    hub_model_id: str = "trl-lib/ddpo-aesthetic-predictor",
    model_filename: str = "aesthetic-model.pth",
):
    scorer = AestheticScorer(
        dtype=dtype,
        model_id=hub_model_id,
        model_filename=model_filename,
    )
    if is_npu_available():
        scorer = scorer.npu()
    elif is_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images: Tensor, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores: Tensor = scorer(images)
        return scores, {}

    return _fn
