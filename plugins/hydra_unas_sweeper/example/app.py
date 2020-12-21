import logging

from typing import Dict

import hydra
from omegaconf import DictConfig

log = logging.getLogger()


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> Dict[str, float]:

    print(cfg)

    return {"val_acc": 0.0}


if __name__ == "__main__":
    main()
