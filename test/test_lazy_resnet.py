from pathlib import Path
from omegaconf import OmegaConf
import torch
import yaml
import os

from hannah.models.resnet.models import ResNet


def test_lazy_resnet_init():
    cwd = os.getcwd()
    config_path = Path(cwd + "/hannah/conf/model/lazy_resnet.yaml")
    with config_path.open("r") as config_file:
        config = yaml.unsafe_load(config_file)
        config = OmegaConf.create(config)
    net = ResNet("resnet", params=config.params, input_shape=[1, 3, 32, 32], labels=config.labels)
    x = torch.randn((3, 3, 32, 32))
    net.sample()
    net.initialize()
    out = net(x)
    assert out.shape == (3, 10)


if __name__ == '__main__':
    test_lazy_resnet_init()