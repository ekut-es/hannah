from speech_recognition.models.qat import (
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBn1d,
    ConvBn2d,
    ConvReLU1d,
    ConvReLU2d,
)
from torch.quantization import default_qconfig
import torch


def test_fused_bn_relu_1d():

    input = torch.rand(8, 1, 3)
    layer = ConvBnReLU1d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.act_fake_quant(output))


def test_fused_bn_relu_2d():
    input = torch.rand(8, 1, 3, 3)
    layer = ConvBnReLU2d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.act_fake_quant(output))


def test_fused_bn_1d():
    input = torch.rand(8, 1, 3)
    layer = ConvBn1d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.act_fake_quant(output))


def test_fused_bn_2d():
    input = torch.rand(8, 1, 3, 3)
    layer = ConvBn2d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.act_fake_quant(output))


def test_fused_relu_1d():
    input = torch.rand(8, 1, 3)
    layer = ConvReLU1d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.act_fake_quant(output))


def test_fused_relu_2d():
    input = torch.rand(8, 1, 3, 3)
    layer = ConvReLU2d(
        in_channels=1, out_channels=1, kernel_size=3, qconfig=default_qconfig
    )
    output = layer(input)
    assert torch.equal(output, layer.act_fake_quant(output))


if __name__ == "__main__":
    test_fused_bn_relu_1d()
    test_fused_bn_relu_2d()

    test_fused_bn_1d()
    test_fused_bn_2d()

    test_fused_relu_1d()
    test_fused_relu_2d()
