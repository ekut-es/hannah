#
# Copyright (c) 2023 Hannah contributors.
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
from collections import namedtuple
from typing import Optional, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.quantization.fake_quantize import FakeQuantize, FakeQuantizeBase
from torch.quantization.observer import (
    MovingAverageMinMaxObserver,
    ObserverBase,
    _with_args,
)

from .rounding import RoundingMode

QConfig = namedtuple("QConfig", ["activation", "weight", "bias", "accumulator"])


class STE(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: Union[Tensor, Parameter],
        quant_function,
    ) -> Tensor:
        ctx.save_for_backward(values)
        quantized_values = quant_function(values)
        return quantized_values

    @staticmethod
    def backward(ctx, grad_outputs):
        (values,) = ctx.saved_tensors
        gate = (torch.abs(values) <= 1.0).float()
        grad_inputs = grad_outputs * gate

        return grad_inputs, None


class QuantizationLoss(nn.Module):
    def __init__(self, bits=8):
        super().__init__()

        self.quantization_loss = torch.tensor([0.0])
        self.scale = torch.nn.Parameter(torch.tensor([1 / 128]))
        self.max = 127

    def forward(self, x):
        self.scale = self.scale.to(x.device)
        self.quantization_loss = self.quantization_loss.to(x.device)
        scaled_x = (
            torch.clamp(torch.round(x / self.scale), -self.max, self.max) * self.scale
        )

        self.quantization_loss = (scaled_x - x).norm(1)

        if not self.training:
            x = scaled_x

        return x


class FixedpointObserver(ObserverBase):
    def __init__(self, observer_quant_min=-128, observer_quant_max=127):
        super().__init__(torch.int32)
        self.qscheme = torch.per_tensor_symmetric
        self.quant_min = observer_quant_min
        self.quant_max = observer_quant_max
        self.device = None

    @torch.jit.export
    def forward(self, x):
        self.device = x.device
        return x

    @torch.jit.export
    def calculate_qparams(self):
        device = self.device
        zero_points = torch.as_tensor([0], device=device)
        quant_factor = float(self.quant_max + 1 - self.quant_min) / 2
        scales = torch.ones(1, device=device) / quant_factor
        return scales, zero_points


class SymmetricQuantization:
    def __init__(
        self, bits: int, rounding_mode: str = "EVEN", debug: bool = False, scale=None
    ) -> None:
        self.bits = bits
        self.max = 2.0 ** (bits - 1) - 1
        self.min = -(2.0 ** (bits - 1))
        if scale is None:
            self.scale = 1.0 / 2 ** (bits - 1)
        else:
            self.scale = scale
        self.rounding_mode = rounding_mode
        self.round = RoundingMode(rounding_mode)
        self.debug = debug

    def quantize(self, x: Union[Tensor, Parameter]) -> Tensor:
        if self.debug:
            print("x", x)
        x = x / self.scale
        x = self.round(x)
        if self.debug:
            print("rounded", x)
        x = torch.clamp(x, self.min, self.max)

        return x

    def __call__(self, x: Union[Tensor, Parameter]) -> Tensor:
        x = self.quantize(x)
        x = x * self.scale

        if self.debug:
            print("fake quantized:", x)

        return x


class PowerOf2Quantization(torch.nn.Module):
    def __init__(self, bits, debug=False):
        super().__init__()
        self.bits = bits
        self.debug = debug

    def quantize(self, x):
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        mask_x = torch.ge(abs_x, 1 / 2 ** ((2**self.bits - 1))).float()

        log_x = torch.ceil(torch.log2(abs_x))

        # This takes care that the number of bits is considered
        # Right now exponent of 0.0 which is the weight 1.0 (2^0.0 = 1.0)
        # is occupied by the weight value 0. But seems to have no negative
        # effect on the contrary this raises the accuracy.
        log_x = torch.clamp(log_x, -(2 ** (self.bits - 1)) + 1, -1.0)
        return log_x * sign_x * mask_x

        return log_x

    def forward(self, x):
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        mask_x = torch.ge(abs_x, 1 / 2 ** ((2**self.bits - 1))).float()

        log_x = torch.ceil(torch.log2(abs_x))

        # This takes care that the number of bits is considered
        # Right now exponent of 0.0 which is the weight 1.0 (2^0.0 = 1.0)
        # is occupied by the weight value 0. But seems to have no negative
        # effect on the contrary this raises the accuracy.
        log_x = torch.clamp(log_x, -(2 ** (self.bits - 1)) + 1, -1.0)

        # This value should match the maximum internal representation of UltraTrail.
        # This is the number of digits after the radix point of WIDE_BW.
        # Currently this is set to 2*(BASE_BW-1) = 14. This can be changed
        # by bw_wide_i in the UltraTrail backend. Therefore the maximum shift is -7.
        # Which achieves quite good results for TC-Res8
        # -7 is equal to use 4 bits for quantization using sign and magnitude represenation.
        # FIXME: Should only be active when UltraTrail is used with correct WIDE_BW
        # log_x = torch.clamp(log_x, -7.0, -1.0)

        x = torch.pow(torch.tensor(2, device=x.device), log_x) * mask_x
        x = x * sign_x
        return x


class STEQuantize(FakeQuantizeBase):
    def __init__(
        self,
        bits: int,
        power_of_2: bool = False,
        noise_prob: float = 1.0,
        rounding_mode: str = "EVEN",
        debug: bool = False,
        dtype: str = "int",
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.bits = bits
        self.noise_prob = noise_prob
        self.rounding_mode = rounding_mode
        self.debug = debug
        self.power_of_2 = power_of_2
        self.dtype = dtype

        if power_of_2:
            self.quantization_function = PowerOf2Quantization(
                bits, scale=scale, debug=self.debug
            )
        else:
            self.quantization_function = SymmetricQuantization(
                bits, rounding_mode=rounding_mode, debug=self.debug
            )

        self.quantization_loss = torch.zeros(1)

    def forward(self, x: Union[Tensor, Parameter]) -> Tensor:
        quantized_x = STE.apply(x, self.quantization_function)
        if self.noise_prob < 1.0 and self.training:
            mask = torch.bernoulli(
                torch.full(x.shape, self.noise_prob, device=x.device)
            ).int()
            reverse_mask = torch.ones(x.shape, device=x.device).int() - mask

            quantized_x = quantized_x * mask + x * reverse_mask

        return quantized_x

    def quantize(self, x):
        return self.quantization_function.quantize(x)

    def calculate_qparams(self):
        raise NotImplementedError(
            "Trainable quantizer has no calculate qparams implementation"
        )

    def extra_repr(self):
        return f"(dtype={self.dtype} bits={self.bits} noise_prob={self.noise_prob}, rounding_mode={self.rounding_mode})"


def get_trax_qat_qconfig(config) -> QConfig:
    bits_bias = config.bw_b if config.bw_b > 0 else config.bw_f
    bits_activation = config.bw_f
    bits_weight = config.bw_w
    bits_accumulator = config.get("bw_acc", 20)
    rounding_mode = config.get("rounding_mode", "EVEN")

    qconfig = QConfig(
        STEQuantize.with_args(
            bits=bits_activation,
            noise_prob=config.get("noise_prob", 1.0),
            rounding_mode=rounding_mode,
        ),
        STEQuantize.with_args(
            bits=bits_weight,
            power_of_2=config.get("power_of_2", False),
            noise_prob=config.get("noise_prob", 1.0),
            rounding_mode=rounding_mode,
            debug=False,
        ),
        STEQuantize.with_args(
            bits=bits_bias,
            noise_prob=config.get("noise_prob", 1.0),
            rounding_mode=rounding_mode,
        ),
        STEQuantize.with_args(
            bits=bits_accumulator,
            noise_prob=config.get("noise_prob", 1.0),
            rounding_mode=rounding_mode,
        ),
    )

    return qconfig
