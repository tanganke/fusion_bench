classnames = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

templates = [
    lambda c: f"a photo of a mango leaf with {c}.",
    lambda c: f"a mango leaf showing symptoms of {c}.",
    lambda c: f"a close-up photo of {c} on a mango leaf.",
    lambda c: f"this mango leaf is affected by {c}.",
    lambda c: f"a mango leaf disease identified as {c}.",
    lambda c: f"a {c} infection on a mango leaf.",
]
