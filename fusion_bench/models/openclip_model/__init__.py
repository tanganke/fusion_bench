import os
import sys


# To make the module importable, we need to add the path to the sys.path
def setup_src():
    import_path = os.path.dirname(__file__)
    if import_path not in sys.path:
        sys.path.append(import_path)


