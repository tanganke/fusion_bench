from datasets import load_dataset


def load_fer2013(path: str = "clip-benchmark/wds_fer2013", split: str = "train"):
    dataset = load_dataset(path, split=split)
    dataset = dataset.remove_columns(["__key__", "__url__"])
    dataset = dataset.rename_columns({"jpg": "image", "cls": "label"})
    return dataset


if __name__ == "__main__":
    dataset = load_fer2013(split="test")
    print(dataset)
