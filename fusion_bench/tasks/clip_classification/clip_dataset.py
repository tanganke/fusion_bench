import torch


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        inputs = self.processor(images=[image], return_tensors="pt")["pixel_values"][0]
        return inputs, item["label"]
