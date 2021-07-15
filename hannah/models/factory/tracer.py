import logging
import math

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
        print(graph_module)
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
            qat.Linear: self._handle_qat_linear,
            qat.LinearReLU: self._handle_qat_linear,
            qconfig.STEQuantize: self._handle_identity,
            torch.nn.ReLU: self._handle_identity,
            torch.nn.Dropout: self._handle_identity,
            torch.nn.Flatten: self._handle_identity,
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
            if input_bits < output_bits:
                output = relay.cast(output, output_dtype)
        return output

    def _handle_identity(self, node, module, result):
        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        self.outputs[node.name] = data
        return None

    def _handle_qat_linear(self, node, module, result):
        inputs = list(node.all_input_nodes)
        data = self.outputs[inputs[0].name]
        self.outputs[node.name] = data
        return None

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

        input_scale = 1 / (2.0 ** 7)
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

        if isinstance(module, qat.ConvBnReLU1d) or isinstance(module, qat.ConvBnReLU2d):
            conv_out = tvm.relay.nn.relu(conv_out)

        if hasattr(module.activation_post_process, "bits"):
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

    def _handle_module(self, node, result):
        print("Handle module", node)
        module = self.modules[node.target]
        if type(module) in self.module_map:
            self.module_map[type(module)](node, module, result)
        else:
            raise Exception(f"Support for module: {module} is not supported")

    def _handle_placeholder(self, node, result):
        var = relay.var(
            node.name, relay.TensorType(result.shape, dtype=self.input_dtype)
        )
        self.outputs[node.name] = var
        self.func_args.append(var)

    def _handle_output(self, node, result):
        inputs = list(node.all_input_nodes)

        for input in inputs:
            self.returns.append(self.outputs[input.name])

    def _handle_function(self, node, result):
        target = node.target
        if target.__name__ == "add":
            inputs = list(node.all_input_nodes)
            lhs = self.outputs[inputs[0].name]
            rhs = self.outputs[inputs[1].name]
            add = tvm.relay.add(lhs, rhs)
            self.outputs[node.name] = add
        elif target.__name__ == "sum":
            inputs = list(node.all_input_nodes)
            data = self.outputs[inputs[0].name]
            sum = tvm.relay.sum(data, axis=2, keepdims=True)
            self.outputs[node.name] = sum
        elif target.__name__ == "truediv":
            inputs = list(node.all_input_nodes)
            data = self.outputs[inputs[0].name]
            div = tvm.relay.sum(data, axis=2, keepdims=True)
            self.outputs[node.name] = div
        else:
            raise Exception(f"Unandled function {target}")

    def run_node(self, node):
        result = super().run_node(node)

        if node.op == "call_module":
            self._handle_module(node, result)
        elif node.op == "call_function":
            self._handle_function(node, result)
        elif node.op == "output":
            self._handle_output(node, result)
        elif node.op == "placeholder":
            self._handle_placeholder(node, result)
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

        return tvm_mod, self.params
