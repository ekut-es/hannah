from pathlib import Path
from omegaconf import OmegaConf
import torch
import yaml
import os

from hannah.models.resnet.models import ResNet


def test_lazy_resnet_init():
    cwd = os.getcwd()
    config_path = Path(cwd + "/hannah/conf/model/lazy_resnet.yaml")
    input_shape = [1, 3, 336, 336]
    with config_path.open("r") as config_file:
        config = yaml.unsafe_load(config_file)
        config = OmegaConf.create(config)
    net = ResNet("resnet", params=config.params, input_shape=input_shape, labels=config.labels)
    x = torch.randn(input_shape)
    net.sample()
    net.initialize()
    out = net(x)
    assert out.shape == (1, 10)


if __name__ == '__main__':
    test_lazy_resnet_init()