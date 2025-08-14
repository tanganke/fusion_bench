"Constants for CLIP Vision Model Merging"

TASK_NAMES_TA8 = [
    "sun397",
    "stanford-cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
]
"The 8 tasks used in the Task Arithmetic paper."
TASK_NAMES_TALL8 = TASK_NAMES_TA8
"The 8 tasks used in the Tall Mask paper"
TASK_NAMES_TALL10 = TASK_NAMES_TA8 + ["oxford_flowers102", "pcam"]
TASK_NAMES_TALL12 = TASK_NAMES_TALL10 + [
    "fer2013",
    "oxford-iiit-pet",
]
TASK_NAMES_TALL14 = TASK_NAMES_TALL12 + [
    "stl10",
    "cifar100",
]
"The 14 tasks used in the TALL mask paper"
TASK_NAMES_TALL16 = TASK_NAMES_TALL14 + ["cifar10", "food101"]
TASK_NAMES_TALL18 = TASK_NAMES_TALL16 + ["fashion_mnist", "emnist_letters"]
TASK_NAMES_TALL20 = TASK_NAMES_TALL18 + ["kmnist", "rendered-sst2"]
"The 20 tasks used in the TALL mask paper"
TASK_NAMES_TA8_CAP = [
    "SUN397",
    "Cars",
    "RESISC45",
    "EuroSAT",
    "SVHN",
    "GTSRB",
    "MNIST",
    "DTD",
]
TASK_NAMES_TALL8_CAP = TASK_NAMES_TA8_CAP
TASK_NAMES_TALL10_CAP = TASK_NAMES_TALL8_CAP + ["Flowers102", "PCAM"]
TASK_NAMES_TALL12_CAP = TASK_NAMES_TALL10_CAP + ["FER2013", "OxfordIIITPet"]
TASK_NAMES_TALL14_CAP = TASK_NAMES_TALL12_CAP + ["STL10", "CIFAR100"]
TASK_NAMES_TALL16_CAP = TASK_NAMES_TALL14_CAP + ["CIFAR10", "Food101"]
TASK_NAMES_TALL18_CAP = TASK_NAMES_TALL16_CAP + ["FashionMNIST", "EMNIST"]
TASK_NAMES_TALL20_CAP = TASK_NAMES_TALL18_CAP + ["KMNIST", "RenderedSST2"]
