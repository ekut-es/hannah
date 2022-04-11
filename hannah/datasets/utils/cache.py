import os

from joblib import Memory


def cachify(func, compress=False):
    CACHE_DIR = os.getenv("HANNAH_CACHE_DIR", None)

    if CACHE_DIR:
        CACHE_SIZE = os.getenv("HANNAH_CACHE_SIZE", None)
        VERBOSE = int(os.getenv("HANNAH_CACHE_VERBOSE", 0))
        cache = Memory(
            location=CACHE_DIR,
            bytes_limit=CACHE_SIZE,
            verbose=VERBOSE,
            compress=compress,
        )
        cached_func = cache.cache(func)
    else:
        cached_func = func

    return cached_func
