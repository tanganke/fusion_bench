import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import lightning as L
import torch

from fusion_bench.utils.pylogger import get_rankzero_logger

log = get_rankzero_logger(__name__)

T = TypeVar("T")


def seed_everything_by_time(fabric: Optional[L.Fabric] = None) -> int:
    """
    Set seed for all processes based on current timestamp.

    This function generates a time-based seed on the global zero process and broadcasts
    it to all other processes in a distributed setting to ensure reproducibility across
    all workers. When no fabric instance is provided, it generates a seed locally without
    synchronization.

    Args:
        fabric: Optional Lightning Fabric instance for distributed synchronization.
                If None, seed is generated locally without broadcasting.

    Returns:
        The seed value used for random number generation.

    Example:
        ```python
        import lightning as L
        from fusion_bench.utils.fabric import seed_everything_by_time

        # With fabric (distributed)
        fabric = L.Fabric(accelerator="auto", devices=2)
        fabric.launch()
        seed = seed_everything_by_time(fabric)
        print(f"All processes using seed: {seed}")

        # Without fabric (single process)
        seed = seed_everything_by_time()
        print(f"Using seed: {seed}")
        ```

    Note:
        - In distributed settings, only the global zero process generates the seed
        - All other processes receive the broadcasted seed for consistency
        - The seed is based on `time.time()`, so it will differ across runs
    """
    # Generate seed on global zero process, None on others
    if fabric is None or fabric.is_global_zero:
        seed = int(time.time())
        log.info(f"Generated time-based seed: {seed}")
    else:
        seed = None

    # Broadcast seed to all processes in distributed setting
    if fabric is not None:
        log.debug(f"Broadcasting seed `{seed}` to all processes")
        fabric.barrier()
        seed = fabric.broadcast(seed, src=0)

    # Apply seed to all random number generators
    L.seed_everything(seed)
    return seed


def is_distributed(fabric: Optional[L.Fabric] = None) -> bool:
    """
    Check if running in distributed mode (multi-process).

    Args:
        fabric: Optional Lightning Fabric instance. If None, returns False.

    Returns:
        True if running with multiple processes, False otherwise.

    Example:
        ```python
        fabric = L.Fabric(accelerator="auto", devices=2)
        fabric.launch()
        if is_distributed(fabric):
            print("Running in distributed mode")
        ```
    """
    return fabric is not None and fabric.world_size > 1


def get_world_info(fabric: Optional[L.Fabric] = None) -> Dict[str, Any]:
    """
    Get comprehensive information about the distributed setup.

    Args:
        fabric: Optional Lightning Fabric instance.

    Returns:
        Dictionary containing:
        - world_size: Total number of processes
        - global_rank: Global rank of current process
        - local_rank: Local rank on current node
        - is_global_zero: Whether this is the main process
        - is_distributed: Whether running in distributed mode

    Example:
        ```python
        fabric = L.Fabric(accelerator="auto", devices=2)
        fabric.launch()
        info = get_world_info(fabric)
        print(f"Process {info['global_rank']}/{info['world_size']}")
        ```
    """
    if fabric is None:
        return {
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
            "is_global_zero": True,
            "is_distributed": False,
        }

    return {
        "world_size": fabric.world_size,
        "global_rank": fabric.global_rank,
        "local_rank": fabric.local_rank,
        "is_global_zero": fabric.is_global_zero,
        "is_distributed": fabric.world_size > 1,
    }


def wait_for_everyone(
    fabric: Optional[L.Fabric] = None, message: Optional[str] = None
) -> None:
    """
    Synchronize all processes with optional logging.

    This is a wrapper around fabric.barrier() with optional message logging.

    Args:
        fabric: Optional Lightning Fabric instance. If None, does nothing.
        message: Optional message to log before synchronization.

    Example:
        ```python
        fabric = L.Fabric(accelerator="auto", devices=2)
        fabric.launch()

        # Do some work...
        wait_for_everyone(fabric, "Waiting after model loading")
        # All processes synchronized
        ```
    """
    if fabric is not None:
        if message and fabric.is_global_zero:
            log.info(message)
        fabric.barrier()


@contextmanager
def rank_zero_only_context(fabric: Optional[L.Fabric] = None):
    """
    Context manager to execute code block only on global rank 0.

    Args:
        fabric: Optional Lightning Fabric instance.

    Example:
        ```python
        fabric = L.Fabric(accelerator="auto", devices=2)
        fabric.launch()

        with rank_zero_only_context(fabric):
            print("This prints only on rank 0")
            save_checkpoint(model, "checkpoint.pt")
        ```
    """
    should_execute = fabric is None or fabric.is_global_zero
    try:
        yield should_execute
    finally:
        pass


def print_on_rank_zero(*args, fabric: Optional[L.Fabric] = None, **kwargs) -> None:
    """
    Print message only on global rank 0.

    Args:
        *args: Arguments to pass to print().
        fabric: Optional Lightning Fabric instance.
        **kwargs: Keyword arguments to pass to print().

    Example:
        ```python
        fabric = L.Fabric(accelerator="auto", devices=2)
        fabric.launch()

        print_on_rank_zero("Starting training", fabric=fabric)
        # Prints only on rank 0
        ```
    """
    if fabric is None or fabric.is_global_zero:
        print(*args, **kwargs)
