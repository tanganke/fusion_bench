test_datasets_20 = {
    "sun397": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/sun397",
        "split": "test",
    },
    "stanford-cars": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/stanford_cars",
        "split": "test",
    },
    "resisc45": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/resisc45",
        "split": "test",
    },
    "eurosat": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/eurosat",
        "split": "test",
    },
    "svhn": {
        "_target_": "datasets.load_dataset",
        "_args_": ["svhn", "cropped_digits"],
        "split": "test",
    },
    "gtsrb": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/gtsrb",
        "split": "test",
    },
    "mnist": {"_target_": "datasets.load_dataset", "path": "mnist", "split": "test"},
    "dtd": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/dtd",
        "split": "test",
    },
    "oxford_flowers102": {
        "_target_": "datasets.load_dataset",
        "path": "dpdl-benchmark/oxford_flowers102",
        "split": "test",
    },
    "pcam": {
        "_target_": "datasets.load_dataset",
        "path": "1aurent/PatchCamelyon",
        "split": "test",
    },
    "fer2013": {
        "_target_": "fusion_bench.dataset.fer2013.load_fer2013",
        "split": "test",
    },
    "oxford-iiit-pet": {
        "_target_": "datasets.load_dataset",
        "path": "timm/oxford-iiit-pet",
        "split": "test",
    },
    "stl10": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/stl10",
        "split": "test",
    },
    "cifar100": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/cifar100",
        "split": "test",
    },
    "cifar10": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/cifar10",
        "split": "test",
    },
    "food101": {
        "_target_": "datasets.load_dataset",
        "path": "ethz/food101",
        "split": "validation",
    },
    "fashion_mnist": {
        "_target_": "datasets.load_dataset",
        "path": "zalando-datasets/fashion_mnist",
        "split": "test",
    },
    "emnist_letters": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/emnist_letters",
        "split": "test",
    },
    "kmnist": {
        "_target_": "datasets.load_dataset",
        "path": "tanganke/kmnist",
        "split": "test",
    },
    "rendered-sst2": {
        "_target_": "datasets.load_dataset",
        "path": "nateraw/rendered-sst2",
        "split": "test",
    },
}
