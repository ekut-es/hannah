import logging
import math

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

    communication_overhead = 0.1 * math.sqrt(len(cfg.cpus)-1)
    cpi = 0.0
    for cpu in cfg.cpus:
        cpi += 1.0 if cpu < 2 else 2.0
    cpi -= communication_overhead
        

    return {
        "val_loss": float(loss),
        "tot_mem": float(mem),
        "tot_weights": float(weights),
        "cpi": float(cpi)
    }


if __name__ == "__main__":
    main()
