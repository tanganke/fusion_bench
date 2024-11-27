from collections import OrderedDict, namedtuple
from functools import wraps
from itertools import chain
from typing import Callable, Optional, TypeVar, cast

import numpy as np
from xxhash import xxh3_64_hexdigest

__all__ = ["np_lru_cache"]

TCallable = TypeVar("TCallable", bound=Callable)

_NpCacheInfo = namedtuple("NpCacheInfo", ["hits", "misses", "maxsize", "currsize"])


def np_lru_cache(
    user_function: TCallable = None, *, maxsize: Optional[int] = 16
) -> TCallable:
    """Wrapper similar to functool's lru_cache, but can handle caching numpy arrays.
    Uses xxhash to hash the raw bytes of the argument array(s) + shape information
    to prevent collisions on arrays with identical data but different dimensions.

    Intentionally has a smaller default maxsize than lru_cache - if you're using this
    wrapper, you are likely trying to avoid some slow computations on large arrays.
    There's no reason to hold onto 128 of these in memory unless you have to.

    Exposes .cache_info and .cache_clear methods, much like lru_cache.

    Does not have the thread-safety features of lru_cache.

    Parameters
    ----------
    user_function : TCallable, optional
    maxsize : int, optional
        Max number of entries in the cache. None for no limit, by default 16

    Returns
    -------
    TCallable
        Wrapped function. Should be mypy-compliant.

    Notes
    ------
    This is implemented similarly to the old lru_cache implementation that
    was discarded for performance upgrades. Generating a hash is by far the
    slowest step of this wrapper, however, so optimizing getting and setting
    the cache is not really going to yield much benefit.

    Does NOT look inside of collections to generate a hash for a number of
    performance-related reasons. For example, this function can be cached:

    >>> fun(np.array([1, 2, 3]), 3, kind="type")

    This one cannot:

    >>> fun(arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])])

    """

    if isinstance(maxsize, int):
        if maxsize < 0:
            maxsize = 0

    def actual_np_cache(user_function):
        # OrderedDict is not threadsafe for updates, but is for reads.
        # The use case for this wrapper is CPU-bound tasks so
        # worrying about thread-safety adds unnecessary overhead
        cache = OrderedDict()
        hits = misses = 0
        cache_len = cache.__len__
        cache_del = cache.popitem
        cache_move_to_end = cache.move_to_end
        full = False

        if maxsize is None:

            @wraps(user_function)
            def _np_cache_wrapper(*args, **kwargs):
                nonlocal hits, misses
                key = _make_hash_key(*args, **kwargs)
                if key not in cache:
                    misses += 1
                    cache[key] = user_function(*args, **kwargs)
                else:
                    hits += 1
                return cache[key]

        elif maxsize == 0:

            @wraps(user_function)
            def _np_cache_wrapper(*args, **kwargs):
                nonlocal misses
                misses += 1
                return user_function(*args, **kwargs)

        else:

            @wraps(user_function)
            def _np_cache_wrapper(*args, **kwargs):
                nonlocal hits, misses, full
                key = _make_hash_key(*args, **kwargs)
                if key not in cache:
                    misses += 1
                    cache[key] = user_function(*args, **kwargs)
                    if full:
                        cache_del(last=False)
                    else:
                        full = cache_len() >= maxsize
                else:
                    hits += 1
                    cache_move_to_end(key)
                return cache[key]

        def cache_info():
            return _NpCacheInfo(hits, misses, maxsize, cache_len())

        def cache_clear():
            nonlocal hits, misses, full
            cache.clear()
            hits = misses = 0
            full = False

        _np_cache_wrapper.cache_info = cache_info
        _np_cache_wrapper.cache_clear = cache_clear
        _np_cache_wrapper.cache = cache

        return _np_cache_wrapper

    if user_function:
        return cast(TCallable, actual_np_cache)(user_function)

    return cast(TCallable, actual_np_cache)


HASH_FUNCTIONS = {np.ndarray: xxh3_64_hexdigest}


def _hasher(obj):
    return HASH_FUNCTIONS.get(type(obj), hash)(obj)


def _make_hash_key(*args, **kwargs):
    """This approach cares about the order of keyword arguments (that is,
    f(arr=a, order="c") will be cached separately from f(order="c", arr=a)

    This is much slower than the equivalent function in functools.lru_cache
    because every element must be inspected to determine if is an array.
    Monkeypatching np.ndarray.__hash__ is unfortunately not possible, but
    would fix this issue."""
    key = tuple(map(_hasher, args))
    if kwargs:
        key += tuple(map(_hasher, chain.from_iterable(kwargs.items())))
    return _HashedArrSeq(key)


class _HashedArrSeq(list):
    """Analogous to _HashedSeq in functools. Essentially caches the __hash__ call
    so that the hash is not recomputed each time the OrderedDict cache interacts
    with this object."""

    __slots__ = "hashvalue"

    def __init__(self, tup):
        self.hashvalue = hash(tup)
        self[:] = [self.hashvalue]

    def __hash__(self):
        return self.hashvalue
