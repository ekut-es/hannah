import logging

from typing import Dict

import hydra
from omegaconf import DictConfig

log = logging.getLogger()


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> Dict[str, float]:

    from pprint import pprint

    pprint(cfg)

    acc = 1.0 / len(cfg.conv_layers)
    mem = len(cfg.conv_layers) * 1.0

    return {"val_acc": acc, "tot_mem": mem}


if __name__ == "__main__":
    main()
