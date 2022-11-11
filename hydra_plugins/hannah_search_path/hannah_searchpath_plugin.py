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
import logging
import pathlib
from typing import Optional

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HannahSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        config_dir = pathlib.Path(".")
        search_path.prepend(provider="hannah", path=f"file://{config_dir}")
        search_path.append(provider="hannah", path="pkg://hannah.conf")

        # Add hannah_tvm to search path
        # FIXME: add generic plugin discovery
        try:
            import hannah_tvm.conf as conf
            import hannah_tvm.config as config  # noqa
        except ModuleNotFoundError:
            logging.debug(
                "Could not find hannah_tvm.conf, tvm backend is not available"
            )
        else:
            search_path.append(provider="hannah_tvm", path="pkg://hannah_tvm.conf")
