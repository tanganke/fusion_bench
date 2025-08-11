from difflib import get_close_matches
from typing import Any, Iterable, List, Optional

__all__ = [
    "first",
    "has_length",
    "join_list",
    "attr_equal",
    "validate_and_suggest_corrections",
]


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


def validate_and_suggest_corrections(
    obj: Any, values: Iterable[Any], *, max_suggestions: int = 3, cutoff: float = 0.6
) -> Any:
    """
    Return *obj* if it is contained in *values*.
    Otherwise raise a helpful ``ValueError`` that lists the closest matches.

    Args:
        obj : Any
            The value to validate.
        values : Iterable[Any]
            The set of allowed values.
        max_suggestions : int, optional
            How many typo-hints to include at most (default 3).
        cutoff : float, optional
            Similarity threshold for suggestions (0.0-1.0, default 0.6).

    Returns:
        The original *obj* if it is valid.

    Raises:
        ValueError: With a friendly message that points out possible typos.
    """
    # Normalise to a list so we can reuse it
    value_list = list(values)

    if obj in value_list:
        return obj

    # Build suggestions
    str_values = list(map(str, value_list))
    matches = get_close_matches(str(obj), str_values, n=max_suggestions, cutoff=cutoff)

    msg = f"Invalid value {obj!r}. Allowed values: {value_list}"
    if matches:
        msg += f". Did you mean {', '.join(repr(m) for m in matches)}?"
    raise ValueError(msg)
