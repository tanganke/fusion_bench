classnames = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

templates = [
    lambda c: f"a photo of a {c} looking face.",
    lambda c: f"a photo of a face showing the emotion: {c}.",
    lambda c: f"a photo of a face looking {c}.",
    lambda c: f"a face that looks {c}.",
    lambda c: f"they look {c}.",
    lambda c: f"look at how {c} they are.",
]
