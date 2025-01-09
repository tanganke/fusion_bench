import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


TASK_TO_LABEL_MAPPING = {
    "sun397": "SUN397",
    "stanford-cars": "Cars",
    "resisc45": "RESISC45",
    "eurosat": "EuroSAT",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "mnist": "MNIST",
    "dtd": "DTD",
    "oxford_flowers102": "Flowers102",
    "pcam": "PCAM",
    "fer2013": "FER2013",
    "oxford-iiit-pet": "OxfordIIITPet",
    "stl10": "STL10",
    "cifar100": "CIFAR100",
    "cifar10": "CIFAR10",
    "food101": "Food101",
    "fashion_mnist": "FashionMNIST",
    "emnist_letters": "EMNIST",
    "kmnist": "KMNIST",
    "rendered-sst2": "RenderedSST2",
}

forest_colors = [
    "#2D5A27",  # Deep forest green
    "#8B9E77",  # Sage
    "#4A6741",  # Moss green
    "#DED29E",  # Dried grass
    "#996B3D",  # Tree bark brown
]
ocean_colors = [
    "#1B4B6B",  # Deep sea blue
    "#4D8FAC",  # Ocean surface
    "#7DB0CD",  # Shallow water
    "#B9D6E8",  # Sea foam
    "#2E6B5E",  # Seaweed green
]

earth_colors = [
    "#8B6F47",  # Rich soil
    "#C7A17C",  # Sandy beige
    "#635147",  # Deep earth
    "#9B8574",  # Clay
    "#4F583D",  # Mountain moss
]

autumn_colors = [
    "#7D4427",  # Russet
    "#B67352",  # Terra cotta
    "#DCA466",  # Golden leaf
    "#8B9E77",  # Faded sage
    "#5B4337",  # Dark bark
]


v1_colors = [
    "#FF4B4B",  # 活力红
    "#FFB03B",  # 明亮黄
    "#3B7FF5",  # 科技蓝
    "#4BC0AA",  # 清新青
    "#9D5BF0",  # 创新紫
]

v2_colors = [
    "#2C73D2",  # 深蓝色（冷色）
    "#FF6B6B",  # 珊瑚红（暖色）
    "#00C9A7",  # 青碧色（冷色）
    "#FF9F43",  # 橙色（暖色）
    "#845EC2",  # 紫色（中性）
]
