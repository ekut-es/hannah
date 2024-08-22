#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pathlib import Path
from omegaconf import OmegaConf
import torch
import yaml
import os

from hannah.models.resnet.models_lazy import ResNet


def test_lazy_resnet_init():
    cwd = os.getcwd()
    config_path = Path(cwd + "/hannah/conf/model/lazy_resnet.yaml")
    input_shape = [1, 3, 336, 336]
    with config_path.open("r") as config_file:
        config = yaml.unsafe_load(config_file)
        config = OmegaConf.create(config)
    net = ResNet(
        "resnet", params=config.params, input_shape=input_shape, labels=config.labels
    )
    x = torch.randn(input_shape)
    net.sample()
    net.initialize()
    out = net(x)
    assert out.shape == (1, 10)


if __name__ == "__main__":
    test_lazy_resnet_init()
