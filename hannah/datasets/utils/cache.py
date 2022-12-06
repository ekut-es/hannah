#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
