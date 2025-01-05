classname_mapping = {
    "0": "T - shirt / top",
    "1": "Trouser",
    "2": "Pullover",
    "3": "Dress",
    "4": "Coat",
    "5": "Sandal",
    "6": "Shirt",
    "7": "Sneaker",
    "8": "Bag",
    "9": "Ankle boot",
}
classnames = [classname_mapping[str(i)] for i in range(10)]

templates = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]
