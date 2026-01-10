from copy import deepcopy
from typing import Iterable, List, Tuple, Union


def dict_get(d: dict, keys: Iterable[str], default=None):
    return [d.get(k, default) for k in keys]


def dict_map(f, d: dict, *, max_level: int = -1, skip_levels=0, inplace=False):
    """Apply function f to each element in dictionary d and return a new dictionary.

    Args:
        f (callable): function to apply
        d (dict): input dictionary
        max_level (int, optional): maximum depth to apply function, -1 means unlimited. Defaults to -1.
        skip_levels (int, optional): number of levels to skip. Defaults to 0.
        inplace (bool, optional): whether to modify input dictionary in place. Defaults to False.

    Returns:
        dict: transformed dictionary
    """
    if not isinstance(d, dict):
        raise TypeError("dict_map: d must be a dict")

    if inplace:
        ans = d
    else:
        ans = deepcopy(d)

    def dict_map_impl(from_dict, to_dict, level):
        if level == max_level:
            return
        for k in from_dict.keys():
            if isinstance(from_dict[k], dict):
                dict_map_impl(from_dict[k], to_dict[k], level + 1)
            else:
                if level < skip_levels:
                    continue
                else:
                    to_dict[k] = f(from_dict[k])

    dict_map_impl(d, ans, 0)
    return ans


def dict_merge(dicts: Iterable[dict], disjoint: bool = True) -> dict:
    """Merge multiple dictionaries into one.

    Args:
        dicts (Iterable[dict]): iterable of dictionaries to merge
        disjoint (bool, optional): if True, raises an error on key conflicts. Defaults to True.

    Returns:
        dict: merged dictionary
    """
    merged_dict = type(dicts[0])()
    for d in dicts:
        for k, v in d.items():
            if disjoint and k in merged_dict:
                raise ValueError(f"Key conflict when merging dictionaries: {k}")
            merged_dict[k] = v
    return merged_dict
