import time
from typing import Optional

import lightning as L

from fusion_bench.utils.pylogger import getRankZeroLogger

log = getRankZeroLogger(__name__)


def seed_everything_by_time(fabric: Optional[L.Fabric] = None):
    """
    Set seed for all processes by time.
    """
    # set seed for all processes
    if fabric is None or fabric.is_global_zero:
        seed = int(time.time())
    else:
        seed = None
    if fabric is not None:
        log.debug(f"Broadcasting seed `{seed}` to all processes")
        fabric.barrier()
        seed = fabric.broadcast(seed, src=0)
    L.seed_everything(seed)
