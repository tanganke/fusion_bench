import os


def path_is_dir_and_not_empty(path: str):
    if path is None:
        return False
    return os.path.isdir(path) and len(os.listdir(path)) > 0
