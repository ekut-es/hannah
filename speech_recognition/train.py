import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from . import conf  # noqa


@hydra.main(config_name="config", config_path="conf")
def main(config=DictConfig):
    validator = instantiate(config.validator, config)
    if config["type"] == "train":
        return validator.train()
    elif config["type"] == "eval":
        return validator.eval()
    elif config["type"] == "eval_vad_keyword":
        logging.error("eval_vad_keyword is not supported at the moment")
    elif config["type"] == "dataset":
        print("Only the dataset will be created and downloaded")
        validator.handleDataset()


if __name__ == "__main__":
    main()
