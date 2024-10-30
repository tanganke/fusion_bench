from typing import Iterable

__all__ = ["first", "has_length"]


def first(iterable: Iterable):
    return next(iter(iterable))


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
