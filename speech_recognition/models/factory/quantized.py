import copy

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import fuse_conv_bn_weights


def _quantize(tensor, qconfig):
    fake_quantized = qconfig(tensor)

    return fake_quantized


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
    ):
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

    def forward(self, input):
        print("Qat conv1d forward")
        print(dir(self))
        print(self.weight)
        output = self.activation_post_process(
            f.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        )

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

    @staticmethod
    def from_float(float_module):
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

        quant_module = Conv1d(
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
            quant_bias = nn.parameter.Parameter(
                _quantize(float_module.bias, float_module.bias_fake_quant)
            )
        quant_module.weight = quant_weight
        quant_module.bias = quant_bias
        quant_module.activation_post_process = copy.deepcopy(
            float_module.activation_post_process
        )

        print(dir(quant_module))

        return quant_module


class ConvReLU1d(nn.Module):
    def __init__():
        super().__init__()

    @staticmethod
    def from_float(float_module):
        return float_module
