from hannah.models.factory.act import DummyActivation
from hannah.models.factory.factory import (
    MajorBlockConfig,
    NetworkFactory,
    ActConfig,
    ELUConfig,
    HardtanhConfig,
    MinorBlockConfig,
    BNConfig,
)

from hannah.models.factory.qat import ConvBnReLU2d

import torch.nn as nn
import torch.quantization as quantization
import torch


def test_act():
    factory = NetworkFactory()

    for config, expected_result in [
        (ELUConfig(), nn.ELU),
        (ActConfig(), nn.ReLU),
        (HardtanhConfig(), nn.Hardtanh),
        (ActConfig("tanh"), nn.Tanh),
        (ActConfig("sigmoid"), nn.Sigmoid),
    ]:
        print(config, expected_result)
        res = factory.act(config)
        assert type(res) == expected_result


def test_minor():
    factory = NetworkFactory()
    factory.default_act = ActConfig("relu")
    factory.default_norm = BNConfig()
    for input_shape, config, expected_result in [
        (
            (128, 8, 16),
            MinorBlockConfig(kernel_size=3, out_channels=16, padding=True),
            nn.Sequential(nn.Conv1d(8, 16, 3, padding=1), DummyActivation()),
        ),
        (
            (128, 8, 16),
            MinorBlockConfig(kernel_size=3, out_channels=16, act=True),
            nn.Sequential(nn.Conv1d(8, 16, 3, padding=1), nn.ReLU()),
        ),
        (
            (128, 8, 16),
            MinorBlockConfig(kernel_size=3, out_channels=16, act=ELUConfig()),
            nn.Sequential(nn.Conv1d(8, 16, 3, padding=1), nn.ELU(alpha=1.0)),
        ),
        (
            (128, 8, 16),
            MinorBlockConfig(
                kernel_size=3, out_channels=16, act=ELUConfig(), norm=True
            ),
            nn.Sequential(
                nn.Conv1d(8, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ELU(alpha=1.0)
            ),
        ),
        (
            (128, 8, 16, 16),
            MinorBlockConfig(
                target="conv2d",
                kernel_size=(3, 3),
                stride=2,
                out_channels=16,
                act=ELUConfig(),
                norm=True,
            ),
            nn.Sequential(
                nn.Conv2d(8, 16, (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ELU(alpha=1.0),
            ),
        ),
        (
            (128, 8, 16, 16),
            MinorBlockConfig(
                target="conv2d",
                kernel_size=(3, 3),
                stride=2,
                dilation=2,
                out_channels=16,
                act=ELUConfig(),
                norm=True,
            ),
            nn.Sequential(
                nn.Conv2d(8, 16, (3, 3), stride=1, padding=1, dilation=2),
                nn.BatchNorm2d(16),
                nn.ELU(alpha=1.0),
            ),
        ),
    ]:
        res_shape, res = factory.minor(input_shape, config)
        print(res, expected_result)

        dummy_input = torch.rand(*input_shape)

        dummy_output = res(dummy_input)
        assert res_shape == dummy_output.shape

        if isinstance(res, nn.Sequential):
            assert len(res) == len(expected_result)
        else:
            res = [res]
            expected_result = [expected_result]

        for res_module, expected_module in zip(res, expected_result):
            assert type(res_module) == type(expected_module)
            if isinstance(res_module, nn.Conv1d):
                assert res_module.kernel_size == expected_module.kernel_size
                assert res_module.in_channels == expected_module.in_channels
                assert res_module.padding == expected_module.padding
                assert res_module.stride == expected_module.stride
                assert res_module.groups == expected_module.groups
                assert res_module.dilation == expected_module.dilation
            elif isinstance(res_module, nn.Conv1d):
                assert res_module.kernel_size == expected_module.kernel_size
                assert res_module.in_channels == expected_module.in_channels
                assert res_module.padding == expected_module.padding
                assert res_module.stride == expected_module.stride
                assert res_module.groups == expected_module.groups
                assert res_module.dilation == expected_module.dilation
            elif isinstance(res_module, nn.BatchNorm1d):
                assert res_module.num_features == expected_module.num_features
                assert res_module.eps == expected_module.eps
                assert res_module.momentum == expected_module.momentum
            elif isinstance(res_module, nn.BatchNorm2d):
                assert res_module.num_features == expected_module.num_features
                assert res_module.eps == expected_module.eps
                assert res_module.momentum == expected_module.momentum


def test_minor_quantized():
    factory = NetworkFactory()
    factory.default_act = ActConfig("relu")
    factory.default_norm = BNConfig()
    factory.default_qconfig = quantization.default_qat_qconfig
    for input_shape, config, expected_result in [
        (
            (128, 8, 16, 16),
            MinorBlockConfig(
                target="conv2d",
                kernel_size=(3, 3),
                stride=2,
                dilation=2,
                out_channels=16,
                act=True,
                norm=True,
            ),
            ConvBnReLU2d(
                8,
                16,
                (3, 3),
                stride=2,
                dilation=2,
                qconfig=quantization.default_qat_qconfig,
            ),
        )
    ]:
        res_shape, res = factory.minor(input_shape, config)
        print(res, expected_result)

        dummy_input = torch.rand(*input_shape)

        dummy_output = res(dummy_input)
        assert res_shape == dummy_output.shape

        if isinstance(res, nn.Sequential):
            assert len(res) == len(expected_result)
        else:
            res = [res]
            expected_result = [expected_result]

        for res_module, expected_module in zip(res, expected_result):
            assert type(res_module) == type(expected_module)
            if isinstance(res_module, nn.Conv1d):
                assert res_module.kernel_size == expected_module.kernel_size
                assert res_module.in_channels == expected_module.in_channels
                assert res_module.padding == expected_module.padding
                assert res_module.stride == expected_module.stride
                assert res_module.groups == expected_module.groups
                assert res_module.dilation == expected_module.dilation
            elif isinstance(res_module, nn.Conv1d):
                assert res_module.kernel_size == expected_module.kernel_size
                assert res_module.in_channels == expected_module.in_channels
                assert res_module.padding == expected_module.padding
                assert res_module.stride == expected_module.stride
                assert res_module.groups == expected_module.groups
                assert res_module.dilation == expected_module.dilation
            elif isinstance(res_module, nn.BatchNorm1d):
                assert res_module.num_features == expected_module.num_features
                assert res_module.eps == expected_module.eps
                assert res_module.momentum == expected_module.momentum
            elif isinstance(res_module, nn.BatchNorm2d):
                assert res_module.num_features == expected_module.num_features
                assert res_module.eps == expected_module.eps
                assert res_module.momentum == expected_module.momentum


def test_major_residual():
    factory = NetworkFactory()
    factory.default_act = ActConfig("relu")
    factory.default_norm = BNConfig()

    config = MajorBlockConfig(
        blocks=[
            MinorBlockConfig(out_channels=32),
            MinorBlockConfig(out_channels=16, stride=2, parallel=True),
        ]
    )
    input_shape = (32, 16, 128)

    output_shape, res_module = factory.residual(input_shape, config)
    input = torch.zeros(*input_shape)
    output = res_module(input)

    assert output_shape == output.shape

    config = MajorBlockConfig(
        blocks=[
            MinorBlockConfig(out_channels=32),
            MinorBlockConfig(out_channels=16, stride=2, parallel=True),
        ],
        reduction="concat",
    )
    input_shape = (32, 16, 128)

    output_shape, res_module = factory.major(input_shape, config)
    input = torch.zeros(*input_shape)
    output = res_module(input)

    assert output_shape == output.shape


if __name__ == "__main__":
    test_act()
    test_minor()
    test_major_residual()
