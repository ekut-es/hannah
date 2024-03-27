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
from abc import abstractmethod, abstractproperty

import torch
from torch.ao.quantization import FixedQParamsFakeQuantize, FixedQParamsObserver

from .data_type import DataType
from .op import Op, Tensor


class BaseQuantize(Op):
    @abstractproperty
    def scale(self):
        ...

    @abstractproperty
    def zero_point(self):
        ...

    @abstractproperty
    def dtype(self):
        ...


class FixedQuantize(BaseQuantize):
    """A fixed quantizer that quantizes the input tensor to a fixed scale and zero point.

    Args:
        scale (float): The scale of the quantized values.
        zero_point (float): The zero point of the quantized values.
        dtype (DataType): The datatype of the quantized values.
    """

    def __init__(self, scale: float, zero_point: float, dtype: DataType):
        super().__init__(self.__class__.__name__)

        self._scale = scale
        self._zero_point = zero_point
        self._dtype = dtype

        range = dtype.range()

        self.observer = FixedQParamsObserver(
            scale=scale,
            zero_point=zero_point,
            dtype=torch.qint8 if dtype.signed else torch.quint8,
            quant_min=range[0],
            quant_max=range[1],
        )
        self.quantizer = FixedQParamsFakeQuantize(observer=self.observer)

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        return self.quantizer(x)

    @property
    def scale(self):
        return self._scale

    @property
    def zero_point(self):
        return self._zero_point

    @property
    def dtype(self):
        return self._dtype
