import logging

from typing import Dict

import hydra
from omegaconf import DictConfig

log = logging.getLogger()


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> Dict[str, float]:

    from pprint import pprint

    pprint(cfg)

    loss = 1.0 / len(cfg.conv_layers) - 0.1 * cfg.lr + 0.1 * cfg.dropout
    mem = 0.0
    old_channels = 1.0
    weights = 0.0
    for layer in cfg.conv_layers:
        local_mem = layer.channels * layer.size * old_channels
        if layer.stride > 0:
            local_mem /= layer.stride
        mem = max(mem, local_mem)
        weights += layer.channels * old_channels * layer.size
        old_channels = layer.channels

    return {
        "val_loss": float(loss),
        "tot_mem": float(mem),
        "tot_weights": float(weights),
    }


if __name__ == "__main__":
    main()
