import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from fusion_bench.utils.validation import validate_file_exists

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem


def save_to_json(obj, path: Union[str, Path], filesystem: "FileSystem" = None):
    """
    save an object to a json file

    Args:
        obj (Any): the object to save
        path (Union[str, Path]): the path to save the object
        filesystem (FileSystem, optional): PyArrow FileSystem to use for writing.
            If None, uses local filesystem via standard Python open().
            Can also be an s3fs.S3FileSystem or fsspec filesystem.
    """
    if filesystem is not None:
        json_str = json.dumps(obj)
        # Check if it's an fsspec-based filesystem (like s3fs)
        if hasattr(filesystem, "open"):
            # Direct fsspec/s3fs usage - more reliable for some endpoints
            path_str = str(path)
            with filesystem.open(path_str, "w") as f:
                f.write(json_str)
        else:
            # Use PyArrow filesystem
            path_str = str(path)
            with filesystem.open_output_stream(path_str) as f:
                f.write(json_str.encode("utf-8"))
    else:
        # Use standard Python file operations
        with open(path, "w") as f:
            json.dump(obj, f)


def load_from_json(
    path: Union[str, Path], filesystem: "FileSystem" = None
) -> Union[dict, list]:
    """load an object from a json file

    Args:
        path (Union[str, Path]): the path to load the object
        filesystem (FileSystem, optional): PyArrow FileSystem to use for reading.
            If None, uses local filesystem via standard Python open().
            Can also be an s3fs.S3FileSystem or fsspec filesystem.

    Returns:
        Union[dict, list]: the loaded object

    Raises:
        ValidationError: If the file doesn't exist (when using local filesystem)
    """
    if filesystem is not None:
        # Check if it's an fsspec-based filesystem (like s3fs)
        if hasattr(filesystem, "open"):
            # Direct fsspec/s3fs usage
            path_str = str(path)
            with filesystem.open(path_str, "r") as f:
                return json.load(f)
        else:
            # Use PyArrow filesystem
            path_str = str(path)
            with filesystem.open_input_stream(path_str) as f:
                json_data = f.read().decode("utf-8")
                return json.loads(json_data)
    else:
        # Use standard Python file operations
        validate_file_exists(path)
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
