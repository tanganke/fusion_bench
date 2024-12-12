classnames_mapping = {
    "0": "お",
    "1": "き",
    "2": "す",
    "3": "つ",
    "4": "な",
    "5": "は",
    "6": "ま",
    "7": "や",
    "8": "れ",
    "9": "を",
}
classnames = [classnames_mapping[str(c)] for c in range(10)]

templates = [
    lambda c: f"a photo of the character {c}.",
]
