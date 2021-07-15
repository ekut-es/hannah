import pytest

import torch
import torch.nn as nn
import tvm

from hannah.models.factory.tracer import QuantizationTracer, RelayConverter
from hannah.models.factory.qat import (
    ConvBn1d,
    ConvBn2d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    Conv1d,
    Conv2d,
)
from hannah.models.factory.qconfig import get_trax_qat_qconfig


class Config:
    bw_b = 8
    bw_f = 8
    bw_w = 8
    power_of2 = False
    rounding_mode = "UPWARD"

    def get(self, name: str, default=None):
        return getattr(self, name, default)


@tvm.relay.transform.function_pass(opt_level=0)
class LegalizeQuantizedTypes(tvm.relay.expr_functor.ExprMutator):
    def __init__(self):
        super().__init__()

        self.dtype_map = {}
        for i in range(1, 9):
            self.dtype_map[f"int{i}"] = "int8"
        for i in range(9, 17):
            self.dtype_map[f"int{i}"] = "int16"
        for i in range(17, 32):
            self.dtype_map[f"int{i}"] = "int32"
        for i in range(33, 65):
            self.dtype_map[f"int{i}"] = "int64"

        for i in range(1, 9):
            self.dtype_map[f"uint{i}"] = "uint8"
        for i in range(9, 17):
            self.dtype_map[f"uint{i}"] = "uint16"
        for i in range(17, 32):
            self.dtype_map[f"uint{i}"] = "uint32"
        for i in range(33, 65):
            self.dtype_map[f"uint{i}"] = "uint64"

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_constant(self, const):
        if const.data.dtype in self.dtype_map:
            return const.astype(self.dtype_map(const.data.dtype))
        return const

    def visit_function(self, fn):
        new_params = []
        binds = {}
        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation
            if isinstance(var_type, tvm.ir.TensorType):
                dtype = var_type.dtype

            # See if we want to replace dtype.
            if dtype in self.dtype_map:
                dtype = self.dtype_map[dtype]
            else:
                dtype = var_type.dtype

            # Generate new variable.
            new_param = tvm.relay.expr.var(
                param.name_hint, shape=var_type.shape, dtype=dtype
            )

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = tvm.relay.expr.bind(new_body, binds)

        # Construct the updated function and return.
        return tvm.relay.Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )

    def visit_call(self, call):
        # print(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        # print(new_args)
        # breakpoint()
        new_attrs = call.attrs
        new_fn = self.visit(call.op)
        new_call = tvm.relay.Call(
            new_fn, new_args, new_attrs, call.type_args, call.span
        )

        if call.op.name == "nn.conv1d":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.conv1d(*new_args, **new_attrs)
        elif call.op.name == "nn.conv2d":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.conv2d(*new_args, **new_attrs)
        elif call.op.name == "nn.conv3d":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.conv3d(*new_args, **new_attrs)
        elif call.op.name == "qnn.requantize":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.qnn.op.requantize(*new_args, **new_attrs)
        elif call.op.name == "cast":
            out_dtype = call.attrs.dtype
            new_attrs = dict(call.attrs)
            new_attrs["dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.cast(*new_args, **new_attrs)
        # print(new_call)

        return new_call


class TestCell(nn.Module):
    def __init__(self, dim=1, act=False):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        if dim == 1:
            self.conv = Conv1d(
                8, 8, 3, qconfig=get_trax_qat_qconfig(Config()), padding=1, bias=True
            )

            self.conv2 = Conv1d(
                8, 8, 3, qconfig=get_trax_qat_qconfig(Config()), padding=1, bias=True
            )
        elif dim == 2:
            self.conv = Conv2d(
                8, 8, 3, qconfig=get_trax_qat_qconfig(Config()), padding=1, bias=True
            )
            self.conv2 = Conv2d(
                8, 8, 3, qconfig=get_trax_qat_qconfig(Config()), padding=1, bias=True
            )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.conv(x)
        x = self.conv2(x)
        return x


@pytest.mark.parametrize("dim,act", [(1, False), (1, True), (2, False), (2, True)])
def test_tracer(dim, act):
    cell = TestCell(dim=dim, act=act)
    print(cell)
    tracer = QuantizationTracer()

    traced_graph = tracer.trace(cell)

    converter = RelayConverter(torch.fx.GraphModule(cell, traced_graph))
    if dim == 1:
        input = torch.rand((1, 8, 32))
    elif dim == 2:
        input = torch.rand((1, 8, 32, 32))

    mod, params = converter.run(input)

    mod = tvm.relay.transform.InferType()(mod)

    mod = LegalizeQuantizedTypes()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    print(mod)
    # print(params)

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = tvm.relay.build(mod, target=target, params=params)

    output_torch = cell(input)
    input_ndarray = (input * 2 ** 7).detach().numpy().astype("byte")

    dev = tvm.device(str(target), 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("x", input_ndarray)
    module.run()
    tvm_output = (
        module.get_output(0, tvm.nd.empty(output_torch.shape, dtype="int8"))
        .numpy()
        .astype(float)
        / 2 ** 7
    )
    print("MSE:   ", ((output_torch.detach().numpy() - tvm_output) ** 2).mean())
    print("MAX_SE:", ((output_torch.detach().numpy() - tvm_output) ** 2).max())
    # print(params)
    # print(lib.lib.get_source())


if __name__ == "__main__":
    test_tracer(1, False)
    test_tracer(2, False)
