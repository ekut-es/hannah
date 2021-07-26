import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.onnx
from pytorch_lightning import Callback

try:
    import onnx  # pytype: disable=import-error
except ModuleNotFoundError:
    onnx = None

try:
    import onnx_tf.backend as tf_backend  # pytype: disable=import-error
except ModuleNotFoundError:
    tf_backend = None

try:
    import onnxruntime.backend as onnxrt_backend  # pytype: disable=import-error
except ModuleNotFoundError:
    onnxrt_backend = None

try:
    import tvm, tvm.relay
except ModuleNotFoundError:
    tvm = None

from ..models.factory.qat import QAT_MODULE_MAPPINGS


def symbolic_batch_dim(model):
    sym_batch_dim = "N"

    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim


class InferenceBackendBase(Callback):
    """ Base class to run val and test on a backend inference engine """

    def __init__(self, val_batches=1, test_batches=1, val_frequency=10):
        self.test_batches = test_batches
        self.val_batches = val_batches
        self.val_frequency = val_frequency
        self.validation_epoch = 0

    def run_batch(self, inputs=None):
        raise NotImplementedError("run_batch is an abstract method")

    def prepare(self, module):
        raise NotImplementedError("prepare is an abstract method")

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.val_batches > 0:
            if self.validation_epoch % self.val_frequency == 0:
                self.prepare(pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx < self.val_batches:
            if self.validation_epoch % self.val_frequency == 0:
                result = self.run_batch(inputs=batch[0])
                target = pl_module.forward(batch[0].to(pl_module.device))

                mse = torch.nn.functional.mse_loss(result, target, reduction="mean")
                pl_module.log("val_backend_mse", mse)
                logging.info("val_backend_mse: %f", mse)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx < self.test_batches:
            result = self.run_batch(inputs=batch[0])
            target = pl_module(batch[0].to(pl_module.device))

            mse = torch.nn.functional.mse_loss(result, target, reduction="mean")
            pl_module.log("test_backend_mse", mse)
            logging.info("test_backend_mse: %f", mse)


class TorchMobileBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        self.script_module = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        self.script_module = model.to_torchscript(method="trace")

    def run_batch(self, inputs=None):
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        return self.script_module(inputs)


class OnnxTFBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=10):
        super(OnnxTFBackend, self).__init__(
            val_batches=val_batches,
            test_batches=test_batches,
            val_frequency=val_frequency,
        )

        self.tf_model = None
        self.interpreter = None

        if onnx is None or tf_backend is None:
            raise Exception(
                "Could not find required libraries for onnx-tf backend please install with poetry instell -E tf-backend"
            )

    def prepare(self, model):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            logging.info("transfering model to onnx")
            dummy_input = model.example_input_array
            torch.onnx.export(model, dummy_input, tmp_dir / "model.onnx", verbose=False)
            logging.info("Creating tf-protobuf")
            onnx_model = onnx.load(tmp_dir / "model.onnx")
            symbolic_batch_dim(onnx_model)
            self.tf_model = tf_backend.prepare(onnx_model)

    def run_batch(self, inputs):
        logging.info("running tf backend on batch")

        result = self.tf_model.run(inputs=inputs)
        result = [torch.from_numpy(res) for res in result]
        return result


class OnnxruntimeBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(
        self, val_batches=1, test_batches=1, val_frequency=10, use_tf_lite=True
    ):
        super(OnnxruntimeBackend, self).__init__(
            val_batches=val_batches, test_batches=test_batches, val_frequency=10
        )

        self.onnxrt_model = None

        if onnx is None or onnxrt_backend is None:
            raise Exception(
                "Could not find required libraries for onnxruntime backend please install with poetry instell -E onnxrt-backend"
            )

    def prepare(self, model):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            logging.info("transfering model to onnx")
            dummy_input = model.example_input_array
            torch.onnx.export(model, dummy_input, tmp_dir / "model.onnx", verbose=False)
            logging.info("Creating onnxrt-model")
            onnx_model = onnx.load(tmp_dir / "model.onnx")
            symbolic_batch_dim(onnx_model)
            self.onnxrt_model = onnxrt_backend.prepare(onnx_model)

    def run_batch(self, inputs=None):
        logging.info("running onnxruntime backend on batch")

        result = self.onnxrt_model.run(inputs=[input.numpy() for input in inputs])
        result = [torch.from_numpy(res) for res in result]
        return result


class TRaxUltraTrailBackend(Callback):
    """TRax UltraTrail backend"""

    def __init__(
        self,
        backend_dir,
        teda_dir,
        standalone,
        rtl_simulation,
        synthesis,
        postsyn_simulation,
        power_estimation,
        num_inferences,
        cols,
        rows,
        period,
        macro_type,
        use_acc_statistic_model,
        use_acc_analytical_model,
        use_acc_teda_data,
    ):
        self.backend_dir = backend_dir
        self.teda_dir = Path(teda_dir)
        self.standalone = standalone
        self.rtl_simulation = rtl_simulation
        self.synthesis = synthesis
        self.postsyn_simulation = postsyn_simulation
        self.power_estimation = power_estimation
        self.num_inferences = num_inferences
        self.bw_w = None  # These are exectracted from models qconfig
        self.bw_b = None
        self.bw_f = None
        self.cols = cols
        self.rows = rows if rows is not None else cols
        self.period = period
        self.macro_type = macro_type
        self.xs = []
        self.ys = []
        self.use_acc_statistic_model = use_acc_statistic_model
        self.use_acc_analytical_model = use_acc_analytical_model
        self.use_acc_teda_data = use_acc_teda_data

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if len(self.xs) < self.num_inferences:
            x = pl_module._extract_features(batch[0].to(pl_module.device))
            x = pl_module.normalizer(x)
            y = pl_module.model(x)

            x = x.cpu().split(1)
            y = y.cpu().split(1)
            y = [t.squeeze() for t in y]

            self.xs.extend(x)
            self.ys.extend(y)

    def _run(self, pl_module):
        # load backend package
        sys.path.append(self.backend_dir)
        from backend.backend import UltraTrailBackend  # pytype: disable=import-error

        classes = pl_module.num_classes
        model = pl_module.model
        mac_mode = "FIXED_POINT"
        if hasattr(model, "qconfig"):
            # Set UltraTrail mac and bit configuration depending on qconfig
            mac_mode = (
                "POWER_OF_TWO"
                if model.qconfig.weight.p.keywords["power_of_2"]
                else "FIXED_POINT"
            )
            self.bw_w = model.qconfig.weight.p.keywords["bits"]
            self.bw_b = model.qconfig.bias.p.keywords["bits"]
            self.bw_f = model.qconfig.activation.p.keywords["bits"]

            # Removing qconfig produces a normal FloatModule
            model = torch.quantization.convert(
                model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        if mac_mode == "POWER_OF_TWO":
            logging.critical(
                "PO2 quantization is enabled. Check that quantization range matches bw_wide_q"
            )

        # execute backend
        backend = UltraTrailBackend(
            teda=self.teda_dir,
            bw_w=self.bw_w,
            bw_b=self.bw_b,
            bw_f=self.bw_f,
            cols=self.cols,
            rows=self.rows,
            period=self.period,
            mac_mode=mac_mode,
            macro_type=self.macro_type,
            classes=classes,
        )

        backend.set_model(
            model.cpu(), pl_module.example_feature_array.cpu(), verbose=True
        )
        backend.set_inputs_and_outputs(self.xs, self.ys)
        backend.prepare()
        if (
            self.use_acc_teda_data
            or self.rtl_simulation
            or self.synthesis
            or self.power_estimation
        ):
            backend.eda(
                self.standalone,
                self.rtl_simulation,
                self.synthesis,
                self.postsyn_simulation,
                self.power_estimation,
            )

        res = backend._do_summary(
            self.use_acc_statistic_model,
            self.use_acc_analytical_model,
            self.use_acc_teda_data,
            self.rtl_simulation,
            self.synthesis,
            self.power_estimation,
        )
        return res

    def estimate(self, pl_module):
        input = pl_module.example_feature_array
        pl_module.eval()
        output = pl_module.model(input)
        self.xs.append(input)
        self.ys.append(output.squeeze())
        return self._run(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        logging.info("Preparing ultratrail")
        res = self._run(pl_module)

        logging.info("Ultratrail metrics")
        for k, v in res.items():
            pl_module.log(k, v)
            logging.info("%s: %s", str(k), str(v))


if tvm is not None:

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


class TVMBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        if tvm is None:
            raise Exception(
                "No tvm installation found please make sure that hannah-tvm is installed"
            )

        self.torch_model = None
        self.model = None
        self.params = None
        self.lib = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        self.torch_model = model

        from ..models.factory.tracer import QuantizationTracer, RelayConverter

        device = model.device

        tracer = QuantizationTracer()

        model.cpu()

        traced_graph = tracer.trace(model.model)
        converter = RelayConverter(torch.fx.GraphModule(model.model, traced_graph))
        mod, params = converter.run(model.example_feature_array)
        mod = tvm.relay.transform.InferType()(mod)
        mod = LegalizeQuantizedTypes()(mod)

        target = "llvm"
        with tvm.transform.PassContext(
            opt_level=3, config={"tir.disable_vectorize": True}
        ):
            lib = tvm.relay.build(mod, target=target, params=params)

        self.model = mod
        self.params = params
        self.lib = lib

        model.to(device)

    def run_batch(self, inputs=None):
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        device = self.torch_model.device
        self.torch_model.cpu()

        feature = self.torch_model.features(inputs)
        feature = self.torch_model.normalizer(feature)

        features = torch.split(feature, 1)

        features = [x.detach().cpu().numpy() for x in features]
        results = []

        for input in features:
            from tvm.contrib import utils
            import numpy as np

            input = input * 128
            input = input.round()
            input = np.clip(input, -128, 127)

            temp = utils.tempdir()
            path_lib = temp.relpath("deploy_lib.tar")
            self.lib.export_library(path_lib)
            print(temp.listdir())

            loaded_lib = tvm.runtime.load_module(path_lib)
            input_data = tvm.nd.array(input)

            module = tvm.contrib.graph_executor.GraphModule(
                loaded_lib["default"](tvm.cpu())
            )
            module.run(data=input_data)
            out_deploy = module.get_output(0).numpy()

            # Print first 10 elements of output
            print(out_deploy.flatten()[0:10])

            out = out_deploy.astype(float)
            out = out / 128

            results.append(torch.from_numpy(out))

        out = torch.stack(results).squeeze(1)

        return out
