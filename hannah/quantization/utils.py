import torch
import torch.autograd as autograd
from hannah.nas.functional_operators.lazy import lazy


class QConfig():
    def __init__(self, weight_bits, activation_bits, per_channel):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.per_channel = per_channel

    def create(self):
        # set parameterized values
        qconfig = {
            "weight": {
                "dtype": "int",
                "bits": lazy(self.weight_bits),
                "method": "symmetric",
                "per_channel": lazy(self.per_channel)
            },
            "activation": {
                "dtype": "int",
                "bits": lazy(self.activation_bits),
                "method": "symmetric",
                "per_channel": False
            }
        }
        return qconfig


def quantize(input, scale, zero_point):
    """
    Range-based Linear Quantization
    """
    return torch.round(torch.div(input, scale) - zero_point)


def dequantize(q_input, scale, zero_point):
    """
    Dequantization of linear-quantized input
    """
    return (q_input + zero_point) * (scale)


def calculate_qparams(
        bits, min_range, max_range,
        mode='symmetric', per_channel=False
):
    """
    Calculate scaling factor and zero-point

    Parameters:
    bits: number of bits for quantization
    min_range: min quantization range
    quant_max: max quantization range
    mode: symmetric or asymmetric quantization
    per_channel: calculate scaling factor per channel
    """

    with torch.no_grad():
        n = 2.0 ** (bits - 1) - 1

        # Symmetric quantization mode
        if per_channel:
            scale, _ = torch.max(
                torch.stack([min_range.abs(), max_range.abs()], dim=1), dim=1
            )
            scale = torch.clamp(scale, min=1e-8) / n
        else:
            scale = max(min_range.abs(), max_range.abs())
            scale = torch.clamp(scale, min=1e-8) / n

        zero_point = torch.tensor(0.)
        # TODO: add asymmetric quantization mode (for activations)

    return scale, zero_point


class SymmetricQuantization(autograd.Function):
    """
    Symmetric quantization of floating-point values,
    given quantization bits and scale.
    """
    @staticmethod
    def forward(ctx, x, bits, scale):
        n = 2.0 ** (bits - 1) - 1
        zero_point = torch.tensor(0.)
        
        # Quantization: scale, round, clamp
        x_q = quantize(x, scale, zero_point)
        x_q = torch.clamp(x_q, -n - 1, n)

        ctx.scale = scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return grad_output.clone() / scale, None, None