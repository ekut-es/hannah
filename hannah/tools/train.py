import logging

from omegaconf import DictConfig

import hydra

from .. import conf  # noqa


@hydra.main(config_name="config", config_path="../conf")
def main(config: DictConfig):

    # Lazily Imported to get quicker tab completion
    from ..train import handle_dataset, nas, train
    from ..utils import log_execution_env_state

    logging.captureWarnings(True)
    try:
        log_execution_env_state()
        if config.get("dataset_creation", None) is not None:
            handle_dataset(config)
        if config.get("nas", None) is not None:
            return nas(config)
        else:
            return train(config)
    except Exception as e:
        logging.exception("Exception Message: %s", str(e))
        raise e


if __name__ == "__main__":
    main()
