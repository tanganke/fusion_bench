import torch

__all__ = ["DUMMY_INPUT_IDS", "DUMMY_LABELS"]

DUMMY_INPUT_IDS = torch.randint(0, 100, (1, 128))
DUMMY_LABELS = torch.randint(0, 100, (1, 128))
