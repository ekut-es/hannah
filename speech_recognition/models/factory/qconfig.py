from collections import namedtuple

import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize, FakeQuantizeBase
from torch.quantization.observer import (MovingAverageMinMaxObserver,
                                         ObserverBase, _with_args)

# FIXME: accumulator is not used at the moment
QConfig = namedtuple("QConfig", ["activation", "weight", "bias"])


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


class TrainableFakeQuantize(FakeQuantizeBase):
    def __init__(self, bits, quantization_loss=True, power_of_2=False, noisy=False):
        super().__init__()

        self.bits = bits

        self.quantization_loss = torch.zeros(1)

    def forward(self, x):

        return x

    def calculate_qparams(self):
        raise NotImplementedError("Power of 2 has no calulate qparams implementation")

    def extra_repr(self):
        return "(bits={self.bits})"


def get_trax_qat_qconfig(config):
    bits_bias = config.bw_b
    bits_activation = config.bw_f
    bits_weight = config.bw_w

    # global_statistics_act = GlobalObserverStatistics()
    # global_statistics_weight = GlobalObserverStatistics()
    # global_statistics_bias = GlobalObserverStatistics()

    qconfig = QConfig(
        FakeQuantize.with_args(
            observer=FixedpointObserver,
            quant_min=-2 ** (bits_activation - 1),
            quant_max=2 ** (bits_activation - 1) - 1,
            observer_quant_min=-2 ** (bits_activation - 1),
            observer_quant_max=2 ** (bits_activation - 1) - 1,
        ),
        FakeQuantize.with_args(
            observer=FixedpointObserver,
            quant_min=-2 ** (bits_weight - 1),
            quant_max=2 ** (bits_weight - 1) - 1,
            observer_quant_min=-2 ** (bits_weight - 1),
            observer_quant_max=2 ** (bits_weight - 1) - 1,
        ),
        FakeQuantize.with_args(
            observer=FixedpointObserver,
            quant_min=-2 ** (bits_bias - 1),
            quant_max=2 ** (bits_bias - 1) - 1,
            observer_quant_min=-2 ** (bits_bias - 1),
            observer_quant_max=2 ** (bits_bias - 1) - 1,
        ),
    )

    # qconfig = QConfig(
    #     QuantizationLoss,
    #     QuantizationLoss,
    #     QuantizationLoss
    # )

    return qconfig
