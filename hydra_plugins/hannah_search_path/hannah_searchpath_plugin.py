import logging
import pathlib
from typing import Optional

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class HannahSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        config_dir = pathlib.Path(".") / "configs"
        if config_dir.exists():
            for path in (x for x in config_dir.iterdir() if x.is_dir()):
                search_path.append(provider="hannah", path=f"file://{path}")

        # Add hannah_tvm to search path
        try:
            import hannah_tvm.conf as conf
            import hannah_tvm.config  # noqa
        except ModuleNotFoundError:
            logging.debug(
                "Could not find hannah_tvm.conf, tvm backend is not available"
            )
        else:
            search_path.append(provider="hannah_tvm", path="pkg://hannah_tvm.conf")
