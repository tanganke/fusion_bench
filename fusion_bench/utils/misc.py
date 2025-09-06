from difflib import get_close_matches
from typing import Any, Iterable, List, Optional, TypeVar, Union

T = TypeVar("T")

__all__ = [
    "first",
    "has_length",
    "join_lists",
    "attr_equal",
    "validate_and_suggest_corrections",
]


def first(iterable: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """
    Return the first element of an iterable.

    Args:
        iterable: The iterable to get the first element from.
        default: The value to return if the iterable is empty. If None and
                the iterable is empty, raises StopIteration.

    Returns:
        The first element of the iterable, or the default value if empty.

    Raises:
        StopIteration: If the iterable is empty and no default is provided.
        TypeError: If the object is not iterable.
    """
    try:
        iterator = iter(iterable)
        return next(iterator)
    except StopIteration:
        if default is not None:
            return default
        raise
    except TypeError as e:
        raise TypeError(
            f"Object of type {type(iterable).__name__} is not iterable"
        ) from e


def has_length(obj: Any) -> bool:
    """
    Check if an object has a length (implements __len__) and len() works correctly.

    Args:
        obj: The object to check for length support.

    Returns:
        bool: True if the object supports len() and doesn't raise an error,
              False otherwise.
    """
    if obj is None:
        return False

    try:
        # Check if __len__ method exists
        if not hasattr(obj, "__len__"):
            return False

        # Try to get the length - this will raise TypeError for unsized objects
        length = len(obj)

        # Verify the length is a non-negative integer
        return isinstance(length, int) and length >= 0
    except (TypeError, AttributeError):
        # TypeError: len() of unsized object
        # AttributeError: if __len__ is not callable somehow
        return False
    except Exception:
        # Any other unexpected error
        return False


def join_lists(list_of_lists: Iterable[Iterable[T]]) -> List[T]:
    """
    Flatten a collection of iterables into a single list.

    Args:
        list_of_lists: An iterable containing iterables to be flattened.

    Returns:
        List[T]: A new list containing all elements from the input iterables
                in order.

    Raises:
        TypeError: If any item in list_of_lists is not iterable.

    Examples:
        >>> join_lists([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
        >>> join_lists([])
        []
        >>> join_lists([[], [1], [], [2, 3]])
        [1, 2, 3]
    """
    if not list_of_lists:
        return []

    result = []
    for i, item in enumerate(list_of_lists):
        try:
            # Check if item is iterable (but not string, which is iterable but
            # usually not what we want to flatten character by character)
            if isinstance(item, (str, bytes)):
                raise TypeError(
                    f"Item at index {i} is a string/bytes, not a list-like iterable"
                )

            # Try to extend with the item
            result.extend(item)
        except TypeError as e:
            if "not iterable" in str(e):
                raise TypeError(
                    f"Item at index {i} (type: {type(item).__name__}) is not iterable"
                ) from e
            else:
                # Re-raise our custom error or other TypeError
                raise

    return result


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
