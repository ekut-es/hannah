import os

from joblib import Memory


def cachify(func):
    CACHE_DIR = os.getenv("HANNAH_CACHE_DIR", None)

    if CACHE_DIR:
        CACHE_SIZE = os.getenv("HANNAH_CACHE_SIZE", None)
        VERBOSE = os.getenv("HANNAH_CACHE_VERBOSE", 0)
        cache = Memory(location=CACHE_DIR, bytes_limit=CACHE_SIZE, verbose=VERBOSE)
        cached_func = cache.cache(func)
    else:
        cached_func = func

    return cached_func
