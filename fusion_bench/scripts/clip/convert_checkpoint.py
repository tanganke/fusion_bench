"""
convert checkpoint from pytorch lightning to huggingface
"""

import argparse
import logging

import torch
from transformers import CLIPProcessor, CLIPVisionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--key", type=str, default="vision_model")
    parser.add_argument("--prefix", type=str, default="vision_model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prefix = args.prefix

    model = CLIPVisionModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # remove the prefix from the keys
    state_dict = {}
    for key in checkpoint[args.key]:
        state_dict[args.prefix + key] = checkpoint[args.key][key]

    model.load_state_dict(state_dict, strict=False)

    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    logging.info(f"converted checkpoint to {args.output}")
