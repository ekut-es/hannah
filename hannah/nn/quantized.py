#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import copy
from typing import Any, Tuple

import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils import fuse_conv_bn_weights

from hannah.quantization.qconfig import STEQuantize


def _quantize(tensor: Parameter, qconfig: STEQuantize) -> Tensor:
    fake_quantized = qconfig(tensor)

    return fake_quantized


class QuantizedConvModule(nn.Module):
    @classmethod
    def from_float(cls, float_module: Any) -> Any:
        assert hasattr(float_module, "weight_fake_quant")
        assert hasattr(float_module, "activation_post_process")

        if hasattr(float_module, "bn"):
            float_module.weight, float_module.bias = fuse_conv_bn_weights(
                float_module.weight,
                float_module.bias,
                float_module.bn.running_mean,
                float_module.bn.running_var,
                float_module.bn.eps,
                float_module.bn.weight,
                float_module.bn.bias,
            )

        quant_module = cls(
            float_module.in_channels,
            float_module.out_channels,
            float_module.kernel_size,
            float_module.stride,
            float_module.padding,
            float_module.dilation,
            float_module.groups,
            float_module.padding_mode,
        )

        quant_weight = nn.parameter.Parameter(
            _quantize(float_module.weight, float_module.weight_fake_quant)
        )
        quant_bias = None
        if float_module.bias is not None:
            if hasattr(float_module, "bias_fake_quant"):
                quant_bias = nn.parameter.Parameter(
                    _quantize(float_module.bias, float_module.bias_fake_quant)
                )
            else:
                quant_bias = nn.parameter.Parameter(
                    _quantize(float_module.bias, float_module.activation_post_process)
                )
        quant_module.weight = quant_weight
        quant_module.bias = quant_bias
        quant_module.activation_post_process = copy.deepcopy(
            float_module.activation_post_process
        )

        return quant_module


class Conv1d(QuantizedConvModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int] = 1,
        padding: Tuple[int] = 0,
        dilation: Tuple[int] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self.bias = None
        self.weight = None
        self.activation_post_process = None

    def _get_name(self):
        return "QuantizedConv1d"

    def forward(self, input: Tensor) -> Tensor:
        output = f.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if hasattr(self, "activation_post_process"):
            output = self.activation_post_process(output)

        return output

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, "
        )  # scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        # if self.dilation != (1,) * len(self.dilation):
        #    s += ', dilation={dilation}'
        # if self.groups != 1:
        #    s += ', groups={groups}'
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class ConvReLU1d(Conv1d):
    def _get_name(self):
        return "QuantizedConvReLU1d"

    def forward(self, input: Tensor) -> Tensor:
        output = f.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        output = f.relu(output)

        if hasattr(self, "activation_post_process"):
            output = self.activation_post_process(output)

        return output


class Conv2d(QuantizedConvModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = 1,
        padding: Tuple[int, int] = 0,
        dilation: Tuple[int, int] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self.bias = None
        self.weight = None
        self.activation_post_process = None

    def _get_name(self):
        return "QuantizedConv1d"

    def forward(self, input: Tensor) -> Tensor:
        output = f.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if hasattr(self, "activation_post_process"):
            output = self.activation_post_process(output)

        return output

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, "
        )  # scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        # if self.dilation != (1,) * len(self.dilation):
        #    s += ', dilation={dilation}'
        # if self.groups != 1:
        #    s += ', groups={groups}'
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class ConvReLU2d(Conv2d):
    def _get_name(self):
        return "QuantizedConvReLU2d"

    def forward(self, input: Tensor) -> Tensor:
        output = f.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        output = f.relu(output)

        if hasattr(self, "activation_post_process"):
            output = self.activation_post_process(output)

        return output


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        self.weight = None
        self.activation_post_process = None

    def _get_name(self):
        return "QuantizedLinear"

    def forward(self, input):
        output = f.linear(input, self.weight, self.bias)

        if hasattr(self, "activation_post_process"):
            output = self.activation_post_process(output)

        return output

    def extra_repr(self):
        s = "{in_features}, {out_features}"
        return s.format(**self.__dict__)

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, "weight_fake_quant")
        assert hasattr(float_module, "activation_post_process")

        if hasattr(float_module, "bn"):
            float_module.weight, float_module.bias = fuse_conv_bn_weights(
                float_module.weight,
                float_module.bias,
                float_module.bn.running_mean,
                float_module.bn.running_var,
                float_module.bn.eps,
                float_module.bn.weight,
                float_module.bn.bias,
            )

        quant_module = cls(float_module.in_features, float_module.out_features)

        quant_weight = nn.parameter.Parameter(
            _quantize(float_module.weight, float_module.weight_fake_quant)
        )
        quant_bias = None
        if float_module.bias is not None:
            if hasattr(float_module, "bias_fake_quant"):
                quant_bias = nn.parameter.Parameter(
                    _quantize(float_module.bias, float_module.bias_fake_quant)
                )
            else:
                quant_bias = nn.parameter.Parameter(
                    _quantize(float_module.bias, float_module.activation_post_process)
                )
        quant_module.weight = quant_weight
        quant_module.bias = quant_bias
        quant_module.activation_post_process = copy.deepcopy(
            float_module.activation_post_process
        )

        return quant_module


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation_post_process = None

    def _get_name(self):
        return "QuantizedIdentity"

    def forward(self, input):
        output = input

        if hasattr(self, "activation_post_process"):
            output = self.activation_post_process(output)

        return output

    def extra_repr(self):
        s = ""
        return s.format(**self.__dict__)

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, "activation_post_process")

        quant_module = cls()

        quant_module.activation_post_process = copy.deepcopy(
            float_module.activation_post_process
        )

        return quant_module
