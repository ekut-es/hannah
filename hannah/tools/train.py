import logging

from omegaconf import DictConfig

import hydra

from .. import conf  # noqa
from ..logo import print_logo
from ..utils import log_execution_env_state


@hydra.main(config_name="config", config_path="../conf", version_base="1.2")
def main(config: DictConfig):
    """

    Args:
      config: DictConfig:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
    logging.captureWarnings(True)
    print_logo()

    from ..train import nas, train

    try:
        log_execution_env_state()
        if config.get("nas", None) is not None:
            return nas(config)
        else:
            return train(config)
    except Exception as e:
        logging.exception("Exception Message: %s", str(e))
        raise e


if __name__ == "__main__":
    main()
