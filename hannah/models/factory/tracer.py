import logging
import math
import copy

from dataclasses import dataclass
from typing import List, Optional

import torch.fx
from tvm.relay import op

from . import qat
from . import qconfig
from . import pooling

try:
    import tvm.relay as relay
    import tvm
except ModuleNotFoundError:
    relay = None
    tvm = None

logger = logging.getLogger("tracer")


@dataclass
class TensorMetadata:
    shape: List[int]
    bits: int

    # Quantization info
    scale: Optional[float] = None
    dtype: Optional[str] = None
    zero_point: Optional[float] = None

    @property
    def relay_dtype(self):
        return f"{self.dtype}{self.bits}"


def parse_dtype(dtype: str):
    if dtype.startswith("int"):
        type = "int"
        bits = int(dtype[3:])
    elif dtype.startswith("float"):
        type = "uint"
        bits = int(dtype[4:])
    elif dtype.startswith("float"):
        type = "float"
        bits = int(dtype[5:])
    else:
        raise Exception(f"Unhandled dtype: {dtype}")
    return type, bits


@tvm.relay.transform.function_pass(opt_level=0)
class LegalizeQuantizedTypes(tvm.relay.expr_functor.ExprMutator):
    def __init__(self):
        super().__init__()

        self.dtype_map = {}
        for i in range(1, 9):
            self.dtype_map[f"int{i}"] = "int8"
        for i in range(9, 17):
            self.dtype_map[f"int{i}"] = "int16"
        for i in range(17, 33):
            self.dtype_map[f"int{i}"] = "int32"
        for i in range(33, 65):
            self.dtype_map[f"int{i}"] = "int64"

        for i in range(1, 9):
            self.dtype_map[f"uint{i}"] = "uint8"
        for i in range(9, 17):
            self.dtype_map[f"uint{i}"] = "uint16"
        for i in range(17, 33):
            self.dtype_map[f"uint{i}"] = "uint32"
        for i in range(33, 65):
            self.dtype_map[f"uint{i}"] = "uint64"

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_constant(self, const):
        if const.data.dtype in self.dtype_map:
            return const.astype(self.dtype_map[const.data.dtype])
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
        elif call.op.name == "nn.dense":
            out_dtype = call.attrs.out_dtype
            new_attrs = dict(call.attrs)
            new_attrs["out_dtype"] = self.dtype_map[out_dtype]
            new_call = tvm.relay.nn.dense(*new_args, **new_attrs)
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


class QuantizationTracer(torch.fx.Tracer):

    LEAF_MODULES = [
        qat.Conv1d,
        qat.Conv2d,
        qat.ConvBn1d,
        qat.ConvBn2d,
        qat.ConvBnReLU1d,
        qat.ConvBnReLU2d,
        qat.ConvReLU1d,
        qat.ConvReLU2d,
        qat.Linear,
        qat.LinearReLU,
        qconfig.STEQuantize,
    ]

    def is_leaf_module(self, module, module_qualified_name):
        for leaf_cls in self.LEAF_MODULES:
            if isinstance(module, leaf_cls):
                return True

        return super().is_leaf_module(module, module_qualified_name)


class RelayConverter(torch.fx.Interpreter):
    def __init__(
        self,
        graph_module,
        input_dtype="int8",
        input_scale=1 / (2 ** 7),
        accumulator_dtype="int20",
    ):
        super().__init__(graph_module)
        self.accumulator_dtype = accumulator_dtype
        self.input_dtype = input_dtype
        self.input_scale = input_scale

        if relay is None:
            raise Exception(
                "TVM does not seem to be installed, please make sure that 'import tvm.relay works'"
            )

        self.tvm_mod = None
        self.modules = {}
        for name, module in graph_module.named_modules():
            self.modules[name] = module

        self.outputs = {}
        self.tensor_info = {}
        self.func_args = []
        self.returns = []
        self.params = {}

        self.module_map = {
            qat.Conv1d: self._handle_qat_conv,
            qat.Conv2d: self._handle_qat_conv,
            qat.ConvBn1d: self._handle_qat_conv,
            qat.ConvBn2d: self._handle_qat_conv,
            qat.ConvBnReLU1d: self._handle_qat_conv,
            qat.ConvBnReLU2d: self._handle_qat_conv,
            qat.ConvReLU1d: self._handle_qat_conv,
            qat.ConvReLU2d: self._handle_qat_conv,
            qat.Linear: self._handle_qat_linear,
            qat.LinearReLU: self._handle_qat_linear,
            qconfig.STEQuantize: self._handle_identity,
            torch.nn.ReLU: self._handle_relu,
            torch.nn.Dropout: self._handle_identity,
            torch.nn.Flatten: self._handle_flatten,
        }

    def _gen_requantize(
        self,
        input,
        input_scale,
        input_dtype,
        output_scale,
        output_dtype,
        use_rescale=False,
        axis=-1,
        rounding="UPWARD",
    ):
        assert input_dtype.startswith("int")
        assert output_dtype.startswith("int")
        if output_dtype == input_dtype and output_scale == input_scale:
            return input

        input_bits = int(input_dtype[3:])
        output_bits = int(output_dtype[3:])

        output = input
        if use_rescale:
            output = tvm.relay.qnn.op.requantize(
                output,
                tvm.relay.const(input_scale),
                tvm.relay.const(0),
                tvm.relay.const(output_scale),
                tvm.relay.const(0),
                axis=axis,
                rounding=rounding,
                out_dtype=output_dtype,
            )
        else:
            rescale = input_scale / output_scale
            rescale_shift = int(math.log2(rescale))
            accumulator_dtype = (
                input_dtype if input_bits > output_bits else output_dtype
            )

            if output_bits > input_bits:
                output = relay.cast(output, output_dtype)
            if rescale != 1.0:
                if 2 ** rescale_shift == rescale:
                    if rescale_shift > 0:
                        output = tvm.relay.left_shift(
                            output,
                            tvm.relay.cast(
                                tvm.relay.const(rescale_shift), dtype=accumulator_dtype
                            ),
                        )
                    else:
                        output = tvm.relay.right_shift(
                            output,
                            tvm.relay.cast(
                                tvm.relay.const(abs(rescale_shift)),
                                dtype=accumulator_dtype,
                            ),
                        )
                else:
                    output = tvm.relay.multiply(
                        output,
                        tvm.relay.cast(
                            tvm.relay.const(int(rescale)), dtype=accumulator_dtype
                        ),
                    )
            if input_bits != output_bits:
                output = relay.cast(output, output_dtype)
        return output

    def _handle_flatten(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        data = self.outputs[inputs[0].name]
        flatten = tvm.relay.nn.batch_flatten(data)
        self.outputs[node.name] = flatten
        output_metadata = copy.deepcopy(self.tensor_info[inputs[0].name])
        output_metadata.shape = result.shape
        self.tensor_info[node.name] = output_metadata

    def _handle_identity(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        data = self.outputs[inputs[0].name]
        self.outputs[node.name] = data
        self.tensor_info[node.name] = self.tensor_info[inputs[0].name]
        return None

    def _handle_qat_linear(self, node, module, result):
        weight = module.weight
        bias = module.bias

        if hasattr(module, "bn"):
            weight, bias = torch.nn.utils.fuse_conv_bn_weights(
                module.weight,
                module.bias,
                module.bn.running_mean,
                module.bn.running_var,
                module.bn.eps,
                module.bn.weight,
                module.bn.bias,
            )

        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        input_info = self.tensor_info[inputs[0].name]
        input_bits = input_info.bits
        input_dtype = input_info.relay_dtype
        input_scale = input_info.scale

        quant_weight = module.weight_fake_quant.quantize(weight)
        quant_bias = module.bias_fake_quant.quantize(bias) if bias is not None else None
        weight_dtype = f"int{module.weight_fake_quant.bits}"
        weight_scale = module.weight_fake_quant.quantization_function.scale

        weight_name = f"{node.name}.weight"
        weight = tvm.relay.Var(
            weight_name, tvm.relay.TensorType(quant_weight.shape, dtype=weight_dtype)
        )
        self.params[weight_name] = tvm.nd.array(
            (quant_weight).detach().numpy().astype("byte")
        )
        if bias is not None:
            bias_dtype = f"int{module.bias_fake_quant.bits}"
            bias_scale = module.bias_fake_quant.quantization_function.scale
            bias_name = f"{node.name}.bias"
            bias = tvm.relay.Var(
                bias_name, tvm.relay.TensorType(quant_bias.shape, dtype=bias_dtype)
            )
            self.params[bias_name] = tvm.nd.array(
                (quant_bias).detach().numpy().astype("byte")
            )

        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]

        linear_out = tvm.relay.nn.dense(
            data, weight, out_dtype=self.accumulator_dtype
        )  # FIXME use proper out dtype

        accumulator_scale = weight_scale * input_scale

        if bias is not None:
            bias = self._gen_requantize(
                bias,
                bias_scale,
                bias_dtype,
                accumulator_scale,
                self.accumulator_dtype,
                use_rescale=True,
                axis=0,
            )
            linear_out = tvm.relay.nn.bias_add(linear_out, bias)

        if isinstance(module, qat.ConvBnReLU1d) or isinstance(module, qat.ConvBnReLU2d):
            linear_out = tvm.relay.nn.relu(linear_out)

        if hasattr(module.activation_post_process, "bits") and module.out_quant:
            output_dtype = f"int{module.activation_post_process.bits}"
            output_scale = module.activation_post_process.quantization_function.scale
        else:
            output_dtype = self.accumulator_dtype
            output_scale = accumulator_scale

        # Calculate shift factors
        linear_out = self._gen_requantize(
            linear_out,
            accumulator_scale,
            self.accumulator_dtype,
            output_scale,
            output_dtype,
            use_rescale=True,
        )

        self.outputs[node.name] = linear_out
        dtype, bits = parse_dtype(output_dtype)
        self.tensor_info[node.name] = TensorMetadata(
            shape=result.shape, dtype=dtype, bits=bits, scale=output_scale, zero_point=0
        )

    def _handle_qat_conv(self, node, module, result):
        weight = module.weight
        bias = module.bias

        if hasattr(module, "bn"):
            weight, bias = torch.nn.utils.fuse_conv_bn_weights(
                module.weight,
                module.bias,
                module.bn.running_mean,
                module.bn.running_var,
                module.bn.eps,
                module.bn.weight,
                module.bn.bias,
            )

        padding = tuple(module.padding)
        stride = tuple(module.stride)
        dilation = tuple(module.dilation)
        groups = module.groups
        out_channels = module.out_channels

        quant_weight = module.weight_fake_quant.quantize(weight)
        quant_bias = module.bias_fake_quant.quantize(bias) if bias is not None else None

        weight_dtype = f"int{module.weight_fake_quant.bits}"
        weight_scale = module.weight_fake_quant.quantization_function.scale

        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        input_info = self.tensor_info[inputs[0].name]
        input_bits = input_info.bits
        input_dtype = input_info.relay_dtype
        input_scale = input_info.scale

        weight_name = f"{node.name}.weight"
        weight = tvm.relay.Var(
            weight_name, tvm.relay.TensorType(quant_weight.shape, dtype=weight_dtype)
        )
        self.params[weight_name] = tvm.nd.array(
            (quant_weight).detach().numpy().astype("byte")
        )
        if bias is not None:
            bias_dtype = f"int{module.bias_fake_quant.bits}"
            bias_scale = module.bias_fake_quant.quantization_function.scale
            bias_name = f"{node.name}.bias"
            bias = tvm.relay.Var(
                bias_name, tvm.relay.TensorType(quant_bias.shape, dtype=bias_dtype)
            )
            self.params[bias_name] = tvm.nd.array(
                (quant_bias).detach().numpy().astype("byte")
            )

        if quant_weight.dim() == 3:
            conv_out = tvm.relay.nn.conv1d(
                data,
                weight,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=out_channels,
                kernel_size=quant_weight.size(2),
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype=self.accumulator_dtype,
            )  # FIXME use proper out dtype
        elif quant_weight.dim() == 4:
            conv_out = tvm.relay.nn.conv2d(
                data,
                weight,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=out_channels,
                kernel_size=(quant_weight.size(2), quant_weight.size(3)),
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype=self.accumulator_dtype,
            )
        else:
            raise Exception(
                f"Quantized weights of dimension {quant_weight.dim()} are not supported"
            )

        accumulator_scale = weight_scale * input_scale

        if bias is not None:
            bias = self._gen_requantize(
                bias,
                bias_scale,
                bias_dtype,
                accumulator_scale,
                self.accumulator_dtype,
                use_rescale=True,
                axis=0,
            )
            conv_out = tvm.relay.nn.bias_add(conv_out, bias)

        if (
            isinstance(module, qat.ConvBnReLU1d)
            or isinstance(module, qat.ConvBnReLU2d)
            or isinstance(module, qat.ConvReLU1d)
            or isinstance(module, qat.ConvReLU2d)
        ):
            conv_out = tvm.relay.nn.relu(conv_out)

        if (
            hasattr(module.activation_post_process, "bits")
            and getattr(module, "out_quant", True) is True
        ):
            output_dtype = f"int{module.activation_post_process.bits}"
            output_scale = module.activation_post_process.quantization_function.scale
        else:
            output_dtype = self.accumulator_dtype
            output_scale = accumulator_scale

        # Calculate shift factors
        conv_out = self._gen_requantize(
            conv_out,
            accumulator_scale,
            self.accumulator_dtype,
            output_scale,
            output_dtype,
            use_rescale=True,
        )

        self.outputs[node.name] = conv_out
        dtype, bits = parse_dtype(output_dtype)
        self.tensor_info[node.name] = TensorMetadata(
            shape=result.shape, dtype=dtype, bits=bits, scale=output_scale, zero_point=0
        )

    def _handle_relu(self, node, module, result):
        inputs = list(node.all_input_nodes)
        assert len(inputs) == 1
        data = self.outputs[inputs[0].name]
        relu = tvm.relay.nn.relu(data)
        self.outputs[node.name] = relu
        output_metadata = copy.deepcopy(self.tensor_info[inputs[0].name])
        self.tensor_info[node.name] = output_metadata

    def _handle_module(self, node, result):
        module = self.modules[node.target]
        if type(module) in self.module_map:
            self.module_map[type(module)](node, module, result)
        else:
            raise Exception(f"Support for module: {module} is not supported")

    def _handle_placeholder(self, node, result):
        print("handle_placeholder", self.input_scale)
        var = relay.var(
            node.name, relay.TensorType(result.shape, dtype=self.input_dtype)
        )
        self.outputs[node.name] = var
        dtype, bits = parse_dtype(self.input_dtype)
        self.tensor_info[node.name] = TensorMetadata(
            shape=result.shape, dtype=dtype, bits=bits, scale=self.input_scale
        )
        self.func_args.append(var)

    def _handle_output(self, node, result):
        inputs = list(node.all_input_nodes)

        for input in inputs:
            self.returns.append(self.outputs[input.name])

    def _handle_function(self, node, result):
        target = node.target

        print(node, target.__name__)
        if target.__name__ == "add":
            inputs = list(node.all_input_nodes)
            assert len(inputs) == 2
            lhs = self.outputs[inputs[0].name]
            rhs = self.outputs[inputs[1].name]
            lhs_data = self.tensor_info[inputs[0].name]
            rhs_data = self.tensor_info[inputs[1].name]
            assert lhs_data.dtype == rhs_data.dtype
            output_dtype = lhs_data.dtype
            output_bits = max(lhs_data.bits, rhs_data.bits)
            output_scale = min(lhs_data.scale, rhs_data.scale)

            lhs = self._gen_requantize(
                lhs,
                lhs_data.scale,
                f"{lhs_data.dtype}{lhs_data.bits}",
                output_scale,
                f"{output_dtype}{output_bits}",
                axis=1,
                use_rescale=True,
            )
            rhs = self._gen_requantize(
                rhs,
                rhs_data.scale,
                f"{rhs_data.dtype}{rhs_data.bits}",
                output_scale,
                f"{output_dtype}{output_bits}",
                axis=1,
                use_rescale=True,
            )

            add = tvm.relay.add(lhs, rhs)
            self.outputs[node.name] = add
            self.tensor_info[node.name] = TensorMetadata(
                shape=result.shape,
                bits=output_bits,
                scale=output_scale,
                zero_point=0,
                dtype=output_dtype,
            )
        elif target.__name__ == "sum":
            inputs = list(node.all_input_nodes)
            assert len(inputs) == 1
            data = self.outputs[inputs[0].name]
            data = tvm.relay.cast(data, self.accumulator_dtype)
            sum = tvm.relay.sum(data, axis=2, keepdims=True)
            self.outputs[node.name] = sum
            self.tensor_info[node.name] = self.tensor_info[inputs[0].name]
        elif target.__name__ == "truediv":
            inputs = list(node.all_input_nodes)
            assert len(inputs) == 1
            data = self.outputs[inputs[0].name]
            div = tvm.relay.cast(data, self.accumulator_dtype) / tvm.relay.cast(
                tvm.relay.const(node.args[1]), self.accumulator_dtype
            )
            self.outputs[node.name] = div
            self.tensor_info[node.name] = self.tensor_info[inputs[0].name]
        else:
            raise Exception(f"Unandled function {target}")

    def run_node(self, node):
        result = super().run_node(node)
        print("node:", node)
        # print(result)

        if node.op == "call_module":
            result_metadata = self._handle_module(node, result)
        elif node.op == "call_function":
            result_metadata = self._handle_function(node, result)
        elif node.op == "output":
            result_metadata = self._handle_output(node, result)
        elif node.op == "placeholder":
            result_metadata = self._handle_placeholder(node, result)
        else:
            raise Exception(f"Node {node} with op {node.op} is not supported")

        return result

    def propagate(self, *args):
        return super().run(*args)

    def run(self, input):
        tvm_mod = tvm.IRModule()

        super().run(input)

        ret = (
            self.returns[0] if len(self.returns) == 1 else tvm.relay.Tuple(self.returns)
        )
        free_vars = relay.analysis.free_vars(ret)
        function = relay.Function(free_vars, ret)
        tvm_mod["main"] = function

        tvm_mod = tvm.relay.transform.InferType()(tvm_mod)
        tvm_mod = LegalizeQuantizedTypes()(tvm_mod)

        return tvm_mod, self.params
