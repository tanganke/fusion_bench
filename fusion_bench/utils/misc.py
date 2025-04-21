from typing import Iterable, List

__all__ = ["first", "has_length", "join_list"]


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


def join_list(list_of_list: List[List]):
    ans = []
    for item in list_of_list:
        ans.extend(item)
    return ans
