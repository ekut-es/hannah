import os
import pathlib

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HannahSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        config_dir = pathlib.Path(".") / "configs"
        for path in (x for x in config_dir.iterdir() if x.is_dir()):
            search_path.append(provider="hannah", path=f"file://{path}")
