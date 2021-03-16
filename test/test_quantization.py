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


def test_fused_conv1d():
    class Config:
        bw_b = 8
        bw_f = 8
        bw_w = 6

    config = Config()
    qconfig = get_trax_qat_qconfig(config)

    class Model(nn.Module):
        def __init__(self, qconfig):
            super().__init__()
            self.qconfig = qconfig
            self.conv = Conv1d(
                in_channels=1, out_channels=1, kernel_size=3, qconfig=qconfig
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    input = torch.rand(8, 1, 3)
    model = Model(qconfig)
    print(model)
    output = model(input)

    quantized_model = convert(model, mapping=QAT_MODULE_MAPPINGS)
    print(quantized_model)

    print(quantized_model.conv.weight)

    quantized_output = quantized_model(input)
    print(output, quantized_output)


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
    test_fused_conv1d()

    # test_fused_bn_relu_1d()
    # test_fused_bn_relu_2d()

    # test_fused_bn_1d()
    # test_fused_bn_2d()

    # test_fused_relu_1d()
    # test_fused_relu_2d()
