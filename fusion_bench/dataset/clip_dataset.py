"""
This module provodes a class to convert dataset whose object is a list of dictionaries with keys "image" and "label" to dataset whose object is a tuple of tensors (inputs, label) for CLIP models.
"""

import torch
from transformers import CLIPProcessor


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor: CLIPProcessor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item, dict):
            item = item
        elif isinstance(item, (tuple, list)):
            assert len(item) == 2, "Each item should be a tuple or list of length 2"
            item = {"image": item[0], "label": item[1]}
        else:
            raise ValueError("Each item should be a dictionary or a tuple of length 2")
        image = item["image"]
        inputs = self.processor(images=[image], return_tensors="pt")["pixel_values"][0]
        return inputs, item["label"]
