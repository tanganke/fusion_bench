from datasets import load_dataset

from .clip_dataset import CLIPDataset


def get_classnames_and_templates(dataset_name: str):
    if dataset_name == "mnist":
        from .mnist import classnames, templates
    elif dataset_name == "tanganke/stanford_cars" or dataset_name == "stanford_cars":
        from .stanford_cars import classnames, templates
    elif dataset_name == "tanganke/gtsrb" or dataset_name == "gtsrb":
        from .gtsrb import classnames, templates
    elif dataset_name == "tanganke/resisc45":
        from .resisc45 import classnames, templates
    elif dataset_name == "tanganke/dtd":
        from .dtd import classnames, templates
    elif dataset_name == "tanganke/eurosat":
        from .eurosat import classnames, templates
    elif dataset_name == "cifar10":
        from .cifar10 import classnames, templates
    elif dataset_name == "svhn":
        from .svhn import classnames, templates
    elif dataset_name == "cifar100":
        from .cifar100 import fine_label as classnames
        from .cifar100 import templates
    elif dataset_name.endswith("tanganke/sun397"):
        from .sun397 import classnames, templates
    elif dataset_name == "nateraw/rendered-sst2":
        from .rendered_sst2 import classnames, templates
    elif dataset_name == "tanganke/stl10":
        from .stl10 import classnames, templates
    elif dataset_name == "dpdl-benchmark/oxford_flowers102":
        from .flower102 import classnames, templates
    elif dataset_name == "timm/oxford-iiit-pet":
        from .oxford_iiit_pet import classnames, templates
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return classnames, templates


def _load_hf_dataset(dataset_name):
    if dataset_name == "svhn":
        return load_dataset(dataset_name, "cropped_digits")
    elif dataset_name == "cifar10":
        dataset = load_dataset(dataset_name)
        dataset = dataset.rename_columns({"img": "image"})
        return dataset
    elif dataset_name == "cifar100":
        dataset = load_dataset(dataset_name)
        dataset = dataset.remove_columns(["coarse_label"]).rename_columns({"img": "image", "fine_label": "label"})
        return dataset
    elif dataset_name == "timm/oxford-iiit-pet":
        dataset = load_dataset(dataset_name)
        dataset = dataset.remove_columns(["image_id", "label_cat_dog"])
        return dataset
    else:
        return load_dataset(dataset_name)


def load_clip_dataset(dataset: str, processor):
    hf_dataset = _load_hf_dataset(dataset)
    return (
        CLIPDataset(hf_dataset["train"], processor),
        CLIPDataset(hf_dataset["test"], processor),
    )
