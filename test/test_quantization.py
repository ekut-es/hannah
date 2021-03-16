import pytest
from speech_recognition.models.factory.qat import (
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBn1d,
    ConvBn2d,
    ConvReLU1d,
    ConvReLU2d,
    Conv1d,
    QAT_MODULE_MAPPINGS,
)
from speech_recognition.models.factory.qconfig import get_trax_qat_qconfig
from torch.quantization import default_qconfig, convert
import torch
import torch.nn as nn


@pytest.mark.parametrize(
    "conv_cls", [(Conv1d), (ConvBn1d), (ConvReLU1d), (ConvBnReLU1d)]
)
def test_fused_conv1d(conv_cls):
    class Config:
        bw_b = 8
        bw_f = 8
        bw_w = 6

    config = Config()
    qconfig = get_trax_qat_qconfig(config)

    conv = conv_cls(in_channels=1, out_channels=1, kernel_size=3, qconfig=qconfig)

    class Model(nn.Module):
        def __init__(self, qconfig, conv):
            super().__init__()
            self.qconfig = qconfig
            self.conv = conv

        def forward(self, x):
            x = self.conv(x)
            return x

    input = torch.rand(8, 1, 3)
    model = Model(qconfig, conv)
    model.eval()
    output = model(input)

    quantized_model = convert(model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=False)

    quantized_output = quantized_model(input)

    assert torch.equal(output, quantized_output)


def test_fused_bn_relu_1d():

    input = torch.rand(8, 1, 3)
    layer = ConvBnReLU1d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.activation_post_process(output))


def test_fused_bn_relu_2d():
    input = torch.rand(8, 1, 3, 3)
    layer = ConvBnReLU2d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.activation_post_process(output))


def test_fused_bn_1d():
    input = torch.rand(8, 1, 3)
    layer = ConvBn1d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.activation_post_process(output))


def test_fused_bn_2d():
    input = torch.rand(8, 1, 3, 3)
    layer = ConvBn2d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.activation_post_process(output))


def test_fused_relu_1d():
    input = torch.rand(8, 1, 3)
    layer = ConvReLU1d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.activation_post_process(output))


def test_fused_relu_2d():
    input = torch.rand(8, 1, 3, 3)
    layer = ConvReLU2d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.activation_post_process(output))


if __name__ == "__main__":
    test_fused_conv1d()

    # test_fused_bn_relu_1d()
    # test_fused_bn_relu_2d()

    # test_fused_bn_1d()
    # test_fused_bn_2d()

    # test_fused_relu_1d()
    # test_fused_relu_2d()
