import importlib

__all__ = ["LIBRARY_PATH"]

LIBRARY_PATH = importlib.import_module("fusion_bench").__path__[0]
