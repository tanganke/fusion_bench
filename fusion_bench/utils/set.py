__all__ = ["union"]


def union(*iters) -> set:
    if len(iters) == 0:
        return set()
    s = set().union(*iters)
    return s
