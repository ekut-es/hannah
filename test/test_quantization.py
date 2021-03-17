import pytest
from torch.quantization.qconfig import get_default_qconfig
from speech_recognition.models.factory.qat import (
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBn1d,
    ConvBn2d,
    ConvReLU1d,
    ConvReLU2d,
    Conv1d,
    Conv2d,
    QAT_MODULE_MAPPINGS,
)
from speech_recognition.models.factory.qconfig import get_trax_qat_qconfig
from torch.quantization import default_qconfig, convert
import torch
import torch.nn as nn


@pytest.mark.parametrize(
    "conv_cls,quant",
    [
        (Conv1d, "trax"),
        (ConvBn1d, "trax"),
        (ConvReLU1d, "trax"),
        (ConvBnReLU1d, "trax"),
        (Conv1d, "fbgemm"),
        (ConvBn1d, "fbgemm"),
        (ConvReLU1d, "fbgemm"),
        (ConvBnReLU1d, "fbgemm"),
    ],
)
def test_quantized_conv1d(conv_cls, quant):
    class Config:
        bw_b = 8
        bw_f = 8
        bw_w = 6

    if quant == "trax":
        config = Config()
        qconfig = get_trax_qat_qconfig(config)
    else:
        qconfig = get_default_qconfig(quant)

    conv = conv_cls(in_channels=1, out_channels=1, kernel_size=3, qconfig=qconfig)

    class Model(nn.Module):
        def __init__(self, qconfig, conv):
            super().__init__()
            self.qconfig = qconfig
            self.conv = conv

        def forward(self, x):
            x = self.conv(x)
            return x

    model = Model(qconfig, conv)

    # Run a few times in training mode to update batch norm statistics
    model.train()
    for _i in range(5):
        input = torch.rand(8, 1, 81)
        model(input)

    input = torch.rand(8, 1, 81)
    model.eval()
    output = model(input)

    quantized_model = convert(model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=False)

    quantized_output = quantized_model(input)

    assert torch.allclose(output, quantized_output)


@pytest.mark.parametrize(
    "conv_cls,quant",
    [
        (Conv2d, "trax"),
        (ConvBn2d, "trax"),
        (ConvReLU2d, "trax"),
        (ConvBnReLU2d, "trax"),
        (Conv2d, "fbgemm"),
        (ConvBn2d, "fbgemm"),
        (ConvReLU2d, "fbgemm"),
        (ConvBnReLU2d, "fbgemm"),
    ],
)
def test_quantized_conv2d(conv_cls, quant):
    class Config:
        bw_b = 8
        bw_f = 8
        bw_w = 6

    if quant == "trax":
        config = Config()
        qconfig = get_trax_qat_qconfig(config)
    else:
        qconfig = get_default_qconfig(quant)

    conv = conv_cls(in_channels=1, out_channels=1, kernel_size=3, qconfig=qconfig)

    class Model(nn.Module):
        def __init__(self, qconfig, conv):
            super().__init__()
            self.qconfig = qconfig
            self.conv = conv

        def forward(self, x):
            x = self.conv(x)
            return x

    model = Model(qconfig, conv)

    # Run a few times in training mode to update batch norm statistics
    model.train()
    for _i in range(30):
        input = torch.rand(8, 1, 81, 81)
        model(input)

    input = torch.rand(8, 1, 81, 81)
    model.eval()
    output = model(input)

    quantized_model = convert(model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=False)

    quantized_output = quantized_model(input)

    assert torch.allclose(output, quantized_output, rtol=1e-5, atol=1e-8)


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
