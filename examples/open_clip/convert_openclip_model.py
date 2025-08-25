# %% imports
import argparse
import os
import sys
from typing import Optional, cast

import open_clip.model
import torch
from src.convert_open_clip_to_hf import copy_vision_model_and_projection
from transformers.models.clip import (
    CLIPConfig,
    CLIPModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)

MODEL_ROOT_PATH = ".cache/task_vectors_checkpoints"


# %% load model
def hf_base_model_name_to_clip_base_model_name(hf_base_model_name: str) -> str:
    _HF_BASE_MODEL_NAME_TO_CLIP_BASE_MODEL_NAME = {
        "clip-vit-base-patch32": "ViT-B-32",
        "clip-vit-base-patch16": "ViT-B-16",
        "clip-vit-large-patch14": "ViT-L-14",
    }
    return _HF_BASE_MODEL_NAME_TO_CLIP_BASE_MODEL_NAME[hf_base_model_name]


def hf_base_model_name_to_base_model_path(hf_base_model_name: str) -> str:
    _HF_BASE_MODEL_NAME_TO_BASE_MODEL_PATH = {
        "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
        "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
        "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
    }
    return _HF_BASE_MODEL_NAME_TO_BASE_MODEL_PATH[hf_base_model_name]


def task_name_to_capital(task_name: str) -> str:
    _NAME_TO_CAPITAL = {
        "sun397": "SUN397",
        "stanford-cars": "Cars",
        "resisc45": "RESISC45",
        "eurosat": "EuroSAT",
        "svhn": "SVHN",
        "gtsrb": "GTSRB",
        "mnist": "MNIST",
        "dtd": "DTD",
    }
    return _NAME_TO_CAPITAL[task_name]


def load_model(
    open_clip_checkpoint_path: str,
):
    model = torch.load(open_clip_checkpoint_path, weights_only=False)
    return model


def parse_args():
    global MODEL_ROOT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_root_path",
        type=str,
        default=MODEL_ROOT_PATH,
        help="path to the model root directory",
    )
    parser.add_argument(
        "--hf_base_model_name",
        type=str,
        required=True,
        help="base model name",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        required=True,
        help="path to save the hf vision model",
    )
    args = parser.parse_args()
    MODEL_ROOT_PATH = args.model_root_path
    return args


def convert_open_clip_vision_model_to_hf_by_path(
    open_clip_checkpoint_path: str, hf_base_model_path: str, save_directory: str
):
    # load open_clip model
    image_encoder = load_model(open_clip_checkpoint_path=open_clip_checkpoint_path)
    pt_model: open_clip.model.CLIP = image_encoder.model.eval()

    # construct hf config and model
    config = CLIPConfig.from_pretrained(hf_base_model_path)
    hf_model = CLIPModel(config).eval()

    # copy open_clip model to hf model
    with torch.no_grad():
        copy_vision_model_and_projection(hf_model, pt_model)
        hf_model.logit_scale = pt_model.logit_scale

        # test hf model
        pixel_values = torch.randn(1, 3, 224, 224)

        hf_image_embed = hf_model.get_image_features(pixel_values)
        pt_image_embed = pt_model.encode_image(pixel_values)

        # print the difference between the two image embeddings and assert they are identical
        print((pt_image_embed - hf_image_embed).sum())
        assert torch.allclose(
            hf_image_embed, pt_image_embed, atol=1e-4
        ), "the image embedding of the two models are not identical"

    # save hf vision model to disk
    hf_vision_config = CLIPVisionConfig.from_pretrained(
        pretrained_model_name_or_path=hf_base_model_path
    )
    hf_vision_model = CLIPVisionModel(hf_vision_config)
    hf_vision_model.vision_model.load_state_dict(hf_model.vision_model.state_dict())
    hf_vision_model.save_pretrained(save_directory)
    return hf_vision_model


def convert_open_clip_vision_model_to_hf(
    hf_base_model_name: str,
    task_name: str,
    save_directory: str,
):
    hf_vision_model = convert_open_clip_vision_model_to_hf_by_path(
        open_clip_checkpoint_path=os.path.join(
            MODEL_ROOT_PATH,
            hf_base_model_name_to_clip_base_model_name(hf_base_model_name),
            task_name_to_capital(task_name),
            "finetuned.pt",
        ),
        hf_base_model_path=hf_base_model_name_to_base_model_path(hf_base_model_name),
        save_directory=save_directory,
    )
    return hf_vision_model


if __name__ == "__main__":
    args = parse_args()
    convert_open_clip_vision_model_to_hf(
        args.hf_base_model_name,
        args.task_name,
        args.save_directory,
    )

# %%
