import os
import pathlib

from omegaconf import OmegaConf

import hannah.conf
import hydra

topdir = pathlib.Path(__file__).parent.absolute() / ".."
config_dir = topdir / "hannah" / "conf"

project_config_dir = topdir / "configs"


def test_parse_configs():
    """This simply tests that all configs are parsable by hydra"""
    for config in config_dir.glob("*.yaml"):
        with hydra.initialize_config_module(
            version_base="1.2", config_module="hannah.conf", job_name="test_config"
        ):
            cfg = hydra.compose(config_name=config.stem)
            print("")
            print("config:", config.stem)
            print(OmegaConf.to_yaml(cfg))

    for config in project_config_dir.glob("*/*.yaml"):
        with hydra.initialize_config_module(
            version_base="1.2", config_module="hannah.conf", job_name="test_config"
        ):
            cfg = hydra.compose(config_name=config.stem)
            print("")
            print("config:", config.stem)
            print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    test_parse_configs()
