import logging
import os
from typing import List

from lightning_utilities.core.rank_zero import rank_zero_only

log = logging.getLogger(__name__)


def path_is_dir_and_not_empty(path: str):
    if path is None:
        return False
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def listdir_fullpath(dir: str) -> List[str]:
    """list directory `dir`, return fullpaths

    Args:
        dir (str): directory name

    Returns:
        List[str]: a list of fullpaths
    """
    assert os.path.isdir(dir), "Argument 'dir' must be a Directory"
    names = os.listdir(dir)
    return [os.path.join(dir, name) for name in names]


@rank_zero_only
def create_symlink(src_dir: str, dst_dir: str, link_name: str = None):
    """
    Creates a symbolic link from src_dir to dst_dir.

    Args:
        src_dir (str): The source directory to link to.
        dst_dir (str): The destination directory where the symlink will be created.
        link_name (str, optional): The name of the symlink. If None, uses the basename of src_dir.

    Raises:
        OSError: If the symbolic link creation fails.
        ValueError: If src_dir does not exist or is not a directory.
    """
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory does not exist: {src_dir}")

    if not os.path.isdir(src_dir):
        raise ValueError(f"Source path is not a directory: {src_dir}")

    # Avoid creating symlink if source and destination are the same
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        log.warning(
            "Source and destination directories are the same, skipping symlink creation"
        )
        return

    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Determine link name
    if link_name is None:
        link_name = os.path.basename(src_dir)

    link_path = os.path.join(dst_dir, link_name)
    # if the link already exists, skip
    if os.path.exists(link_path):
        log.warning(f"Symbolic link already exists, skipping: {link_path}")
        return

    try:
        # if the system is windows, use the `mklink` command in "CMD" to create the symlink
        if os.name == "nt":
            os.system(
                f"mklink /J {os.path.abspath(link_path)} {os.path.abspath(src_dir)}"
            )
        else:
            os.symlink(
                src_dir,
                link_path,
                target_is_directory=True,
            )
        log.info(f"Created symbolic link: {link_path} -> {src_dir}")
    except OSError as e:
        log.warning(f"Failed to create symbolic link: {e}")
        raise
