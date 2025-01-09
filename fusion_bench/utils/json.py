import json
from pathlib import Path
from typing import Any, Union


def save_to_json(obj, path: Union[str, Path]):
    """
    save an object to a json file

    Args:
        obj (Any): the object to save
        path (Union[str, Path]): the path to save the object
    """
    with open(path, "w") as f:
        json.dump(obj, f)


def load_from_json(path: Union[str, Path]) -> Union[dict, list]:
    """load an object from a json file

    Args:
        path (Union[str, Path]): the path to load the object

    Returns:
        dict: the loaded object
    """
    with open(path, "r") as f:
        return json.load(f)


def _is_list_of_dict(obj) -> bool:
    if not isinstance(obj, list):
        return False
    for i in obj:
        if not isinstance(i, dict):
            return False
    return True


def _sprint_json_entry(obj):
    if isinstance(obj, str):
        return "string"
    elif isinstance(obj, float):
        return "float"
    elif isinstance(obj, int):
        return "int"
    elif isinstance(obj, list):
        if len(obj) > 0:
            return f"list[{_sprint_json_entry(obj[0])}]"
        else:
            return "list"
    else:
        return type(obj)


def print_json(j: dict, indent="  ", verbose: bool = False, print_type: bool = True):
    R"""print an overview of json file

    Examples:
        >>> print_json(open('path_to_json', 'r'))

    Args:
        j (dict): loaded json file
        indent (str, optional): Defaults to '  '.
    """

    def _print_json(j: dict, level):
        def _sprint(s):
            return indent * level + s

        for k in j.keys():
            if isinstance(j[k], dict):
                print(_sprint(k) + ":")
                _print_json(j[k], level + 1)
            elif _is_list_of_dict(j[k]):
                if verbose:
                    print(_sprint(k) + ": [")
                    for i in range(len(j[k]) - 1):
                        _print_json(j[k][0], level + 2)
                        print(_sprint(f"{indent},"))
                    _print_json(j[k][-1], level + 2)
                    print(_sprint(f"{indent}]"))
                else:
                    print(_sprint(k) + ": [")
                    _print_json(j[k][0], level + 2)
                    print(_sprint(f"{indent}] ... {len(j[k]) - 1} more"))
            else:
                if print_type:
                    print(f"{_sprint(k)}: {_sprint_json_entry(j[k])}")
                else:
                    print(f"{_sprint(k)}: {j[k]}")

    _print_json(j, level=0)
