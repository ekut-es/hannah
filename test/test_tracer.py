import pytest

import torch
import torch.nn as nn
import tvm

from hannah.models.factory.tracer import (
    QuantizationTracer,
    RelayConverter,
    LegalizeQuantizedTypes,
)
from hannah.models.factory.qat import (
    ConvBn1d,
    ConvBn2d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvReLU1d,
    ConvReLU2d,
    Conv1d,
    Conv2d,
    Linear,
)
from hannah.models.factory.pooling import ApproximateGlobalAveragePooling1D
from hannah.models.factory.reduction import ReductionBlockAdd
from hannah.models.factory.qconfig import get_trax_qat_qconfig


class Config:
    bw_b = 4
    bw_f = 8
    bw_w = 8
    power_of2 = False
    rounding_mode = "UPWARD"

    def get(self, name: str, default=None):
        return getattr(self, name, default)


class TestCell(nn.Module):
    def __init__(self, dim=1, act=False):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        if dim == 1:
            if act:
                self.conv = ConvBnReLU1d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv = ConvBn1d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
            if act:
                self.conv2 = ConvReLU1d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv2 = Conv1d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
        elif dim == 2:
            if act:
                self.conv = ConvBnReLU2d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv = ConvBn2d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
            if act:
                self.conv2 = ConvReLU2d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv2 = Conv2d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=True,
                )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.conv(x)
        x = self.conv2(x)
        return x


def run_test(cell, input_shape, act, input_bits, output_bits, out_dtype):
    print(cell)
    cell.eval()
    tracer = QuantizationTracer()

    traced_graph = tracer.trace(cell)

    converter = RelayConverter(torch.fx.GraphModule(cell, traced_graph))

    input = torch.rand(input_shape)

    mod, params = converter.run(input)
    print(mod)

    mod = tvm.relay.transform.InferType()(mod)

    mod = LegalizeQuantizedTypes()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    print(mod)
    # print(params)

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = tvm.relay.build(mod, target=target, params=params)

    output_torch = cell(input)
    input_ndarray = (input * 2 ** (input_bits - 1)).detach().numpy().astype("byte")

    dev = tvm.device(str(target), 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("x", input_ndarray)
    module.run()
    output_scale = 1 / 2 ** (output_bits - 1)
    print("Output scale:", output_scale)
    tvm_output = (
        module.get_output(0, tvm.nd.empty(output_torch.shape, dtype=out_dtype))
        .numpy()
        .astype(float)
        * output_scale
    )

    mse = ((output_torch.detach().numpy() - tvm_output) ** 2).mean()
    max_se = ((output_torch.detach().numpy() - tvm_output) ** 2).max()

    print(output_torch)
    print(tvm_output)

    print("MSE:   ", mse)
    print("MAX_SE:", max_se)

    assert mse < 1 / (2 ** (min(output_bits, input_bits) - 1))
    assert max_se < 2 / (2 ** (min(output_bits, input_bits) - 1))


@pytest.mark.parametrize("dim,act", [(1, False), (1, True), (2, False), (2, True)])
def test_tracer(dim, act):
    cell = TestCell(dim=dim, act=act)
    input_bits = 8
    output_bits = 8

    if dim == 1:
        input_shape = (1, 8, 32)
    elif dim == 2:
        input_shape = (1, 8, 32, 32)

    run_test(cell, input_shape, act, input_bits, output_bits, "int8")

    # print(params)
    # print(lib.lib.get_source())


class TestCellReduction(nn.Module):
    def __init__(self, dim=1, act=False):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        if dim == 1:
            conv = Conv1d(
                8,
                8,
                3,
                qconfig=get_trax_qat_qconfig(Config()),
                padding=1,
                bias=True,
                out_quant=False,
            )

            conv2 = Conv1d(
                8, 8, 3, qconfig=get_trax_qat_qconfig(Config()), padding=1, bias=True
            )
        elif dim == 2:
            conv = Conv2d(
                8,
                8,
                3,
                qconfig=get_trax_qat_qconfig(Config()),
                padding=1,
                bias=True,
                out_quant=False,
            )
            conv2 = Conv2d(
                8, 8, 3, qconfig=get_trax_qat_qconfig(Config()), padding=1, bias=True
            )
        self.red = ReductionBlockAdd(conv, conv2)

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.red(x)
        return x


def test_tracer_reduction(dim=1, act=True):
    cell = TestCellReduction(dim=dim, act=act)
    if dim == 1:
        input_shape = (1, 8, 32)
    elif dim == 2:
        input_shape = (1, 8, 32, 32)

    run_test(cell, input_shape, act, 8, 15, "int32")


class TestCellLinear(nn.Module):
    def __init__(self, act=False):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        if act:
            self.linear = LinearReLU(
                128, 32, bias=False, qconfig=get_trax_qat_qconfig(Config())
            )
        else:
            self.linear = Linear(
                128, 32, bias=False, qconfig=get_trax_qat_qconfig(Config())
            )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.linear(x)
        x = self.activation_post_process(x)
        return x


def test_tracer_linear(act=False):
    cell = TestCellLinear()
    input_shape = (1, 128)
    act = True
    run_test(cell, input_shape, act, 8, 8, "int8")


class TestCellPooling(nn.Module):
    def __init__(self, act=False):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        self.pool = ApproximateGlobalAveragePooling1D(
            16, qconfig=get_trax_qat_qconfig(Config())
        )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.pool(x)
        return x


def test_tracer_pooling():
    cell = TestCellPooling()
    input_shape = (1, 64, 16)
    act = False
    input_bits = 8
    output_bits = 8
    out_dtype = "int32"
    run_test(cell, input_shape, act, input_bits, output_bits, out_dtype)


if __name__ == "__main__":
    test_tracer(1, True)
    # test_tracer(2, True)
    # test_tracer_reduction()
    test_tracer_linear(True)
    # test_tracer_pooling()
