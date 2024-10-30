import os

def path_is_dir_and_not_empty(path: str):
    return os.path.isdir(path) and len(os.listdir(path)) > 0
