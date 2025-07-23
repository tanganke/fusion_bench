from typing import Iterable, List

__all__ = ["first", "has_length", "join_list", "attr_equal"]


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


def attr_equal(obj, attr: str, value):
    """
    Check if the attribute of the object is equal to the given value.
    Returns False if the attribute does not exist or is not equal to the value.

    Args:
        obj: The object to check.
        attr (str): The attribute name to check.
        value: The value to compare against.

    Returns:
        bool: True if the attribute exists and is equal to the value, False otherwise.
    """
    if not hasattr(obj, attr):
        return False
    return getattr(obj, attr) == value
