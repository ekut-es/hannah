from collections import namedtuple

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize, FakeQuantizeBase
from torch.quantization.observer import (
    MovingAverageMinMaxObserver,
    ObserverBase,
    _with_args,
)

# FIXME: accumulator is not used at the moment
QConfig = namedtuple("QConfig", ["activation", "weight", "bias"])


class STE(autograd.Function):
    @staticmethod
    def forward(ctx, values, quant_function):
        ctx.save_for_backward(values)
        quantized_values = quant_function(values)
        return quantized_values

    @staticmethod
    def backward(ctx, grad_outputs):
        # print("grad_outputs:", grad_outputs)
        values, = ctx.saved_tensors
        gate = (torch.abs(values) <= 1).float()
        grad_inputs = grad_outputs * gate
        # print("grad_inputs", grad_inputs)

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
    def __init__(self, bits, debug=False):
        self.bits = bits
        self.max = 2.0 ** (bits - 1) - 1
        self.min = -2.0 ** (bits - 1)
        self.scale = 1.0 / 2 ** (bits - 1)
        self.debug = debug

    def __call__(self, x):
        if self.debug:
            print("x", x)
        x = x / self.scale
        x = torch.round(x)
        if self.debug:
            print("rounded", x)
        x = torch.clamp(x, self.min, self.max)
        x = x * self.scale
        if self.debug:
            print("fake quantized:", x)

        return x


class PowerOf2Quantization:
    def __init__(self, bits, debug=False):
        self.bits = bits
        self.debug = debug

    def __call__(self, x):
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        mask_x = torch.ge(abs_x, 1 / 2 ** ((2 ** self.bits - 1))).float()

        log_x = torch.ceil(torch.log2(abs_x))
        log_x = torch.clamp(log_x, -2 ** (self.bits - 1) + 2, 0.0)
        x = torch.pow(torch.tensor(2, device=x.device), log_x) * mask_x
        x = x * sign_x
        return x


class TrainableFakeQuantize(FakeQuantizeBase):
    def __init__(
        self,
        bits,
        quantization_loss=True,
        power_of_2=False,
        noise_prob=1.0,
        debug=False,
    ):
        super().__init__()

        self.bits = bits
        self.noise_prob = noise_prob
        self.debug = debug

        if power_of_2:
            self.quantization_function = PowerOf2Quantization(bits, debug=self.debug)
        else:
            self.quantization_function = SymmetricQuantization(bits, debug=self.debug)

        self.quantization_loss = torch.zeros(1)

    def forward(self, x):

        quantized_x = STE.apply(x, self.quantization_function)
        if self.noise_prob < 1.0 and self.training:
            mask = torch.bernoulli(
                torch.full(x.shape, self.noise_prob, device=x.device)
            ).int()
            reverse_mask = torch.ones(x.shape, device=x.device).int() - mask

            quantized_x = quantized_x * mask + x * reverse_mask

        return quantized_x

    def calculate_qparams(self):
        raise NotImplementedError(
            "Trainable quantizer has no calulate qparams implementation"
        )

    def extra_repr(self):
        return f"(bits={self.bits} noise_prob={self.noise_prob}, )"


def get_trax_qat_qconfig(config):
    bits_bias = config.bw_b
    bits_activation = config.bw_f
    bits_weight = config.bw_w

    qconfig = QConfig(
        TrainableFakeQuantize.with_args(
            bits=bits_activation, noise_prob=config.get("noise_prob", 1.0)
        ),
        TrainableFakeQuantize.with_args(
            bits=bits_weight,
            power_of_2=config.get("power_of_2", True),
            noise_prob=config.get("noise_prob", 1.0),
            debug=False,
        ),
        TrainableFakeQuantize.with_args(
            bits=bits_bias, noise_prob=config.get("noise_prob", 1.0)
        ),
    )

    return qconfig
