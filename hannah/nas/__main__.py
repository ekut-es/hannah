import hydra

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import hannah.conf  # noqa


@hydra.main(config_name="config_darts_nas", config_path="../conf")
def nas(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    nas_trainer = instantiate(config.nas, parent_config=config, _recursive_=False)
    nas_trainer.run()


if __name__ == "__main__":
    nas()
