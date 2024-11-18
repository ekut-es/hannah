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
import torch

from hannah.modules.vision.image_classifier import ImageClassifierModule
from hannah.quantization.callback import QuantizationCallback


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def test_quant_callback():
    callback = QuantizationCallback()

    model = MyModel()

    module = ImageClassifierModule(
        None,
        model,
        None,
        None,
        None,
    )

    module.example_feature_array = torch.randn(1, 3, 224, 224)

    callback.on_fit_start(None, module)

    quantized_model = module.model

    assert isinstance(
        quantized_model.conv, torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d
    )


if __name__ == "__main__":
    test_quant_callback()
