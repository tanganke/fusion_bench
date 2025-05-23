from pathlib import Path
from typing import Literal

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
OPENCLIP_CACHEDIR = Path(Path.home(), "openclip-cachedir", "open_clip").as_posix()
CACHEDIR = None

ALL_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
    "SUN397",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "CIFAR100",
    "PCAM",
    "FER2013",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "EMNIST",
    "KMNIST",
]

DATASETS_8 = ALL_DATASETS[:8]
DATASETS_14 = ALL_DATASETS[:14]
DATASETS_20 = ALL_DATASETS[:20]


def cleanup_dataset_name(dataset_name: str):
    return dataset_name.replace("Val", "") + "Val"


def get_zeroshot_path(root, dataset, model):
    return Path(
        root, model, cleanup_dataset_name(dataset), f"nonlinear_zeroshot.pt"
    ).as_posix()


def get_finetuned_path(root, dataset, model):
    return Path(
        root, model, cleanup_dataset_name(dataset), f"nonlinear_finetuned.pt"
    ).as_posix()


def get_single_task_accuracies_path(model):
    return Path(
        "results/single_task", model, f"nonlinear_ft_accuracies.json"
    ).as_posix()
