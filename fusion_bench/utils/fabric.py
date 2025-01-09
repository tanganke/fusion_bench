import time

import lightning as L


def seed_everything_by_time(fabric: L.Fabric):
    """
    Set seed for all processes by time.
    """
    # set seed for all processes
    if fabric.is_global_zero:
        seed = int(time.time())
    else:
        seed = None
    fabric.barrier()
    seed = fabric.broadcast(seed, src=0)
    L.seed_everything(seed)
