from dataclasses import dataclass, field
from collections import OrderedDict
from numpy.core.fromnumeric import shape
from numpy.core.numeric import count_nonzero
from numpy.core.shape_base import block
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import _piecewise_dispatcher
from tvm.ir.module import IRModule
from tvm.relay.op.nn.nn import conv1d, layer_norm
import yaml
import numpy as np
from tvm.relay import nn
from tvm import auto_scheduler, relay
import tvm
from tvm.contrib import graph_executor


from tvm.relay.frontend.common import infer_shape, infer_type

from tvm.relay.qnn.op import quantize, dequantize, requantize

from tvm.auto_scheduler.task_scheduler import TaskSchedulerCallback
import os
import yaml
import re


class LayerSpace:
    """
    Arbitrary layer with various parameter choices
    like dtype and bit-width
    """

    def __init__(self, dtype=["float32"], repeat=[1], **kwargs) -> None:
        self.space_map = OrderedDict()
        self.space_map["dtype"] = list(dtype)
        self.repeat = repeat
        self.type = "Layer"

    def dims(self):
        config = []
        config.append(len(self.repeat))
        for k, v in self.space_map.items():
            config.append(len(v))
        return config

    def __repr__(self) -> str:
        return (
            self.type
            + ": "
            + " ".join(["{}: {}".format(k, v) for k, v in self.space_map.items()])
        )


class Conv1dSpace(LayerSpace):
    """
    1D-Convolution layer space
    """

    def __init__(
        self,
        kernel_size: list,
        channels: list = [1],
        dtype: list = ["float32"],
        weight_dtype: list = ["float32"],
        repeat: int = [1],
        **kwargs
    ) -> None:
        super().__init__(dtype=dtype, repeat=repeat, **kwargs)
        self.space_map["kernel_size"] = list(kernel_size)
        self.space_map["channels"] = list(channels)
        self.space_map["weight_dtype"] = list(weight_dtype)

        for name, values in kwargs.items():
            self.space_map[name] = list(values)
        self.type = "conv1d"

    def __repr__(self) -> str:
        return "Conv1D{}: ".format(self.repeat) + " ".join(
            ["{}: {}".format(k, v) for k, v in self.space_map.items()]
        )


class DenseSpace(LayerSpace):
    """
    Fully-Connected/Dense layer space
    """

    def __init__(
        self,
        labels,
        repeat=[1],
        units=[None],
        flatten_input=True,
        weight_dtype: list = ["float32"],
        dtype=["float32"],
        **kwargs
    ) -> None:
        super().__init__(dtype=dtype, repeat=repeat, **kwargs)
        self.type = "dense"
        self.space_map["units"] = units
        self.space_map["labels"] = labels

        self.space_map["weight_dtype"] = list(weight_dtype)

        # TODO: check whether necessary
        self.flatten_input = flatten_input

        for name, values in kwargs.items():
            self.space_map[name] = list(values)

    def __repr__(self) -> str:
        return "Dense{}: ".format(self.repeat) + " ".join(
            ["{}: {}".format(k, v) for k, v in self.space_map.items()]
        )


class Pool1dSpace(LayerSpace):
    """
    1D Pooling layer space. Currently only avg pooling
    """

    def __init__(self, repeat=[1], pool_size=[1], strides=[1], **kwargs) -> None:
        super().__init__(repeat=repeat, **kwargs)
        self.type = "pool1d"
        self.space_map["pool_size"] = pool_size
        self.space_map["strides"] = strides

    def __repr__(self) -> str:
        return "Pool1d{}: ".format(self.repeat) + " ".join(
            ["{}: {}".format(k, v) for k, v in self.space_map.items()]
        )


class ActivationSpace(LayerSpace):
    """
    Activation function space:
        act_fun is a list of functions that are (individually) applied to input
    """

    def __init__(self, act_fun, dtype=["float32"], repeat=[1], **kwargs) -> None:
        super().__init__(dtype=dtype, repeat=repeat, **kwargs)
        self.type = "activation"
        self.space_map["act"] = act_fun


class BatchNormSpace(LayerSpace):
    """
    Batch Normalization space.
        fold_bn: indicates whether the bn is folded into the convolution
    """

    def __init__(self, fold_bn=True, dtype=["float32"], repeat=[1], **kwargs) -> None:
        super().__init__(dtype=dtype, repeat=repeat, **kwargs)
        self.type = "batch_norm"
        self.space_map["fold_bn"] = fold_bn


class BlockSpace:
    """
    sequence of layers
    """

    def __init__(self, layers: list, repeat=[1], residual=[]) -> None:
        self.layers = layers
        self.length = len(self.layers)
        self.residual = residual
        self.repeat = repeat

    def dims(self):
        config = []
        config.append(len(self.repeat))
        for layer in self.layers:
            config.extend(layer.dims())
        if self.residual:
            config.extend(self.residual.dims())

        return config

    def __repr__(self) -> str:
        return (
            "Block{}:\n".format(self.repeat)
            + "".join(["\t{}\n".format(l) for l in self.layers])
            + "\n\tResidual:\n\t\t{}".format(self.residual)
        )


class NetworkSpace:
    def __init__(self) -> None:
        self.blocks = []

    def add_layers(self, layers, repeat=[1]):
        block = BlockSpace(layers=layers, repeat=repeat)
        self.blocks.append(block)

    def add_block(self, block):
        self.blocks.append(block)

    def dims(self):
        config = []
        for block in self.blocks:
            config.extend(block.dims())
        return config

    def collapsed_dims(self):
        dims = self.dims()
        collapsed_dims = [d for d in dims if d > 1]
        return collapsed_dims

    def expand_config(self, config):
        dims = self.dims()
        collapsed_dims = self.collapsed_dims()
        expanded_cfg = []
        counter = 0
        for d in dims:
            if d > 1:
                expanded_cfg.append(config[counter])
                counter += 1
            else:
                expanded_cfg.append(0)
        return expanded_cfg

    def from_yaml(self, filepath):
        with open(filepath) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        self.from_dict(data)

    def from_dict(self, dict):
        blocks = []
        for b in dict:
            print(b)
            layers = []
            for l in b["block"]:
                print(l)
                for key in l.keys():
                    print("key:", key)
                    layer_space = SPACE_DICT[key]
                    layer = layer_space(**l[key][0])
                    layers.append(layer)

            if "residual" in b:
                res_layer = []
                for r in b["residual"]:
                    for l in r["block"]:
                        for key in l.keys():
                            layer_space = SPACE_DICT[key]
                            layer = layer_space(**l[key][0])
                            res_layer.append(layer)
                res_repeat = r.get("repeat", [1])
                residual = BlockSpace(layers=res_layer, repeat=res_repeat)
            else:
                residual = []

            block_repeat = b.get("repeat", [1])
            block = BlockSpace(layers=layers, repeat=block_repeat, residual=residual)
            self.add_block(block)

    def __len__(self):
        return np.prod(self.dims())

    def __repr__(self) -> str:
        return "".join(["{}\n".format(b) for b in self.blocks])


class LayerEntity:
    def __init__(self, space, cfg, id=-1) -> None:
        self.id = id
        self.entity_map = OrderedDict()
        self.type = space.type
        assert len(cfg) == len(space.space_map)
        for i, (k, v) in enumerate(space.space_map.items()):
            self.entity_map[k] = v[cfg[i]]

        self.dtype = self.entity_map["dtype"]
        if self.dtype not in SUPPORTED_DTYPES:
            raise InvalidConfigurationError(
                "{} not a supported dtype".format(self.dtype)
            )

    def to_relay(self, input, **kwargs):
        # print("Convert: {} {}".format(self.type, self.id))
        input_dtype = infer_type(input).checked_type.dtype
        # self.entity_map['input_shape'] = infer_type(input).checked_type.shape[-1]
        layer_dtype = self.dtype
        quant_params = kwargs["quant_params"][self.id]
        op = self.quant(
            input,
            input_dtype=input_dtype,
            output_dtype=layer_dtype,
            quant_params=quant_params,
        )
        return op

    def quant(self, input, input_dtype, output_dtype, quant_params):
        if input_dtype != output_dtype:
            if input_dtype == "float32":
                op = quantize(
                    input,
                    output_scale=relay.const(np.float32(quant_params["output_scale"])),
                    output_zero_point=relay.const(
                        np.int32(quant_params["output_zero_point"])
                    ),
                    out_dtype=output_dtype,
                )
            elif output_dtype == "float32":
                op = dequantize(
                    input,
                    input_scale=relay.const(np.float32(quant_params["input_scale"])),
                    input_zero_point=relay.const(
                        np.int32(quant_params["input_zero_point"])
                    ),
                )
            else:
                op = requantize(
                    input,
                    input_scale=relay.const(np.float32(quant_params["input_scale"])),
                    input_zero_point=relay.const(
                        np.int32(quant_params["input_zero_point"])
                    ),
                    output_scale=relay.const(np.float32(quant_params["output_scale"])),
                    output_zero_point=relay.const(
                        np.int32(quant_params["output_zero_point"])
                    ),
                    out_dtype=output_dtype,
                )

        else:
            op = input
        return op

    # def batch_norm_relay(self, op, weight, input_dtype, input_scale, input_zero_point, output_dtype, output_scale, output_zero_point):
    #     if self.entity_map.get('batch_norm', False):
    #         w_shape = infer_shape(weight)[0]
    #         # print('w_shape: ',w_shape)
    #         i_shape = infer_shape(op)[1]
    #         # print("i_shape", i_shape)

    #         if not self.entity_map.get('fold_batch_norm', True):
    #             gamma = relay.var('bn_{}_gamma'.format(self.id), shape=(w_shape,))
    #             beta = relay.var('bn_{}_beta'.format(self.id), shape=(w_shape,))
    #             moving_mean = relay.var('bn_{}_moving_mean'.format(self.id), shape=(w_shape,))
    #             moving_var = relay.var('bn_{}_moving_var'.format(self.id), shape=(w_shape,))

    #             if input_dtype != 'float32':
    #                 op = dequantize(op, input_scale, input_zero_point)
    #             op = nn.batch_norm(op, gamma, beta, moving_mean, moving_var)[0]
    #             if output_dtype != 'float32':
    #                 op = quantize(op, output_scale, output_zero_point)
    #         else:
    #             # TODO: Make bias dtype dependent on input
    #             bias = relay.var(str(self.id) + '_bias', shape=(i_shape,), dtype=input_dtype)
    #             op = nn.bias_add(op, bias)
    #     return op

    def __repr__(self) -> str:
        return (
            self.type
            + " "
            + " ".join(["{}: {}".format(k, v) for k, v in self.entity_map.items()])
        )


class Conv1dEntity(LayerEntity):
    def __init__(self, space, cfg, id=-1) -> None:
        super().__init__(space, cfg, id)
        self.weight_dtype = self.entity_map["weight_dtype"]

    def to_relay(self, input, **kwargs):
        op = super().to_relay(input, **kwargs)
        assert "weight" in kwargs, "Must provide weight for conv1d" + self.id
        layer_dtype = self.dtype

        weight = kwargs["weight"]
        # currently only supports weight quantization from float32 -> int8
        if weight.type_annotation.dtype != self.dtype:
            quant_params = kwargs["quant_params"][self.id]
            weight = quantize(
                weight,
                output_scale=relay.const(np.float32(quant_params["weight_scale"])),
                output_zero_point=relay.const(
                    np.int32(quant_params["weight_zero_point"])
                ),
                out_dtype=self.dtype,
            )
        conv1d_param = {
            "strides": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "channels": None,
            "kernel_size": None,
            "data_layout": "NCW",
            "kernel_layout": "OIW",
            "out_layout": "",
            "out_dtype": layer_dtype,
        }

        for k, v in self.entity_map.items():
            if k in conv1d_param:
                conv1d_param[k] = v
        op = nn.conv1d(op, weight, **conv1d_param)

        return op


class DenseEntity(LayerEntity):
    def __init__(self, space, cfg, flatten_input=True, id=-1) -> None:
        super().__init__(space, cfg, id)
        self.weight_dtype = self.entity_map["weight_dtype"]
        self.flatten_input = flatten_input

    def to_relay(self, data, **kwargs):
        op = super().to_relay(data, **kwargs)

        layer_dtype = self.dtype
        dense_param = {"units": None, "out_dtype": layer_dtype}

        for k, v in self.entity_map.items():
            if k in dense_param:
                dense_param[k] = v

        if self.flatten_input:
            data = relay.reshape(data, newshape=(1, -1))

        weight = kwargs["weight"]
        if weight.type_annotation.dtype != self.dtype:
            quant_params = kwargs["quant_params"][self.id]
            weight = quantize(
                weight,
                output_scale=relay.const(np.float32(quant_params["weight_scale"])),
                output_zero_point=relay.const(
                    np.int32(quant_params["weight_zero_point"])
                ),
                out_dtype=self.dtype,
            )
        op = nn.dense(data, weight, **dense_param)
        return op


class Pool1dEntity(LayerEntity):
    def __init__(self, space, cfg, id) -> None:
        super().__init__(space, cfg, id=id)

    def to_relay(self, data, **kwargs):
        # op = data
        op = super().to_relay(data, **kwargs)
        pool_size = (int(self.entity_map.get("pool_size", 1)),)
        strides = (int(self.entity_map.get("strides", 1)),)
        padding = int(self.entity_map.get("padding", 1))
        op = nn.avg_pool1d(op, pool_size=pool_size, strides=strides, padding=padding)

        return op


class ActivationEntity(LayerEntity):
    def __init__(self, space, cfg, id) -> None:
        super().__init__(space, cfg, id=id)

    def to_relay(self, input, **kwargs):
        op = super().to_relay(input, **kwargs)
        if isinstance(self.entity_map["act"], str):
            act = ACT_DICT[self.entity_map["act"]]
        else:
            act = self.entity_map["act"]
        return act(op)


class BatchNormEntity(LayerEntity):
    def __init__(self, space, cfg, id) -> None:
        super().__init__(space, cfg, id=id)

    def to_relay(self, input, **kwargs):
        op = super().to_relay(input, **kwargs)
        if self.entity_map.get("fold_bn", True):
            i_shape = infer_shape(input)[1]
            i_type = infer_type(input).checked_type.dtype
            bias = relay.var(str(self.id) + "_bias", shape=(i_shape,), dtype=self.dtype)
            op = nn.bias_add(op, bias)
        elif self.dtype != "float32":
            raise InvalidConfigurationError(
                "Batch norm layer with quantization must be folded."
            )
        return op


class BlockEntity:
    def __init__(self, space, cfg, num_prev_layers=0):
        self.layers = []
        num_layers = num_prev_layers
        n = 0
        for j, layer_space in enumerate(space.layers):
            layer_space_len = (
                len(layer_space.space_map) + 1
            )  # +1 bc of repeat param in layers
            layer_config = cfg[n : n + layer_space_len]
            for _ in range(layer_space.repeat[layer_config[0]]):
                new_layer_type = ENTITY_DICT[layer_space.type]
                self.layers.append(
                    new_layer_type(layer_space, layer_config[1:], id=num_layers)
                )
                num_layers += 1
            n += layer_space_len
        self.residual = []

        if space.residual:  # and isinstance(space.residual, BlockSpace):
            self.residual = []
            res_space_len = len(space.residual.dims())
            for _ in range(space.residual.repeat[cfg[n]]):
                self.residual.extend(
                    BlockEntity(
                        space.residual,
                        cfg[n + 1 : n + 1 + res_space_len],
                        num_prev_layers=num_layers,
                    ).layers
                )

    def to_relay(self, input, input_channel, **kwargs):
        i_channels = input_channel
        block_input = input

        # regular block
        op = self.to_relay_layers(self.layers, input, i_channels, **kwargs)
        if self.residual:
            # residual
            input = block_input
            i_channels = input_channel
            res = self.to_relay_layers(self.residual, input, i_channels, **kwargs)

            main_branch_dtype = infer_type(op).checked_type.dtype
            res_branch_dtype = infer_type(res).checked_type.dtype

            if res_branch_dtype != main_branch_dtype:
                quant_params = kwargs["quant_params"][
                    "add_" + str(self.layers[-1].id) + "_" + str(self.residual[-1].id)
                ]
                if main_branch_dtype == "float32":

                    res = dequantize(
                        res,
                        input_scale=relay.const(
                            np.float32(quant_params["input_scale"])
                        ),
                        input_zero_point=relay.const(
                            np.int32(quant_params["input_zero_point"])
                        ),
                    )
                elif res_branch_dtype == "float32":
                    res = quantize(
                        res,
                        output_scale=relay.const(
                            np.float32(quant_params["output_scale"])
                        ),
                        output_zero_point=relay.const(
                            np.int32(quant_params["output_zero_point"])
                        ),
                        out_dtype=main_branch_dtype,
                    )
                else:
                    res = requantize(
                        res,
                        input_scale=relay.const(
                            np.float32(quant_params["input_scale"])
                        ),
                        input_zero_point=relay.const(
                            np.int32(quant_params["input_zero_point"])
                        ),
                        output_scale=relay.const(
                            np.float32(quant_params["output_scale"])
                        ),
                        output_zero_point=relay.const(
                            np.int32(quant_params["output_zero_point"])
                        ),
                        out_dtype=main_branch_dtype,
                    )
            op = relay.add(op, res)
        return op

    # TODO: Remove i_channels everywhere
    def to_relay_layers(self, layers, input, i_channels, **kwargs):
        inp = input
        for i, layer in enumerate(layers):
            if layer.type == "conv1d":
                o_channels = layer.entity_map["channels"]
                i_channels = infer_shape(inp)[1]
                kernel_size = layer.entity_map["kernel_size"]
                weight_dtype = layer.entity_map.get("weight_dtype", "float32")
                weight = relay.var(
                    layer.type + "_weight_{}".format(layer.id),
                    shape=(o_channels, i_channels, kernel_size),
                    dtype=weight_dtype,
                )
                # i_channels = o_channels
            elif layer.type == "dense":
                shape = infer_shape(inp)
                s = shape[1] * shape[2]
                # print("shape", s)
                weight = relay.var(
                    layer.type + "_weight_{}".format(layer.id),
                    shape=(layer.entity_map["labels"], s),
                    dtype=layer.weight_dtype,
                )
            else:
                weight = None
            relay_layer = layer.to_relay(inp, weight=weight, **kwargs)
            inp = relay_layer
        return inp

    def __repr__(self) -> str:
        return (
            "Block:\n"
            + "".join(["\t{}\n".format(l) for l in self.layers])
            + "\n\tResidual:\n"
            + "".join(["\t\t{}\n".format(l) for l in self.residual])
        )


class NetworkEntity:
    def __init__(self, space, cfg) -> None:
        self.blocks = []
        n = 0
        num_layers = 0
        for block_space in space.blocks:
            block_space_len = len(block_space.dims())
            block_config = cfg[n : n + block_space_len]
            for _ in range(block_space.repeat[block_config[0]]):
                self.blocks.append(
                    BlockEntity(
                        block_space, block_config[1:], num_prev_layers=num_layers
                    )
                )
                num_layers += len(self.blocks[-1].layers)
                num_layers += len(self.blocks[-1].residual)
            n += block_space_len

    def to_relay(self, input, **kwargs):
        relay_prog = None
        block_id = 0
        i_channel = input.type_annotation.shape[1]
        for block in self.blocks:
            relay_prog = block.to_relay(input, i_channel, **kwargs)
            block_id += 1
            input = relay_prog

            if "channels" in block.layers[-1].entity_map:
                i_channel = block.layers[-1].entity_map["channels"]
        mod = IRModule.from_expr(relay_prog)
        return mod

    def to_dict(self):
        blocks = []
        for block in self.blocks:
            bl = {"block": [], "residual": []}
            for layer in block.layers:
                bl["block"].append(str(layer))
            for layer in block.residual:
                bl["residual"].append(str(layer))
            blocks.append(bl)
        return blocks

    def quantization_sequence(self, input):
        input_dtype = input.type_annotation.dtype
        type_dict = {}
        layers = []
        for block in self.blocks:
            block_layers = list(layers)
            for layer in block.layers:
                if hasattr(layer, "weight_dtype"):
                    weight_dtype = layer.weight_dtype
                else:
                    weight_dtype = None
                if block_layers:
                    type_dict[layer.id] = (
                        block_layers[-1].dtype,
                        layer.dtype,
                        weight_dtype,
                    )
                else:
                    type_dict[layer.id] = (input_dtype, layer.dtype, weight_dtype)
                block_layers.append(layer)

            res_layers = layers
            for layer in block.residual:
                if hasattr(layer, "weight_dtype"):
                    weight_dtype = layer.weight_dtype
                else:
                    weight_dtype = None
                if res_layers:
                    type_dict[layer.id] = (
                        res_layers[-1].dtype,
                        layer.dtype,
                        weight_dtype,
                    )
                else:
                    type_dict[layer.id] = (input_dtype, layer.dtype, weight_dtype)
                res_layers.append(layer)
            # layers = block_layers# [-len(block.layers):] + res_layers[-len(block.residual):]

            if block.residual:
                # type_dict['add_' + str(block_layers[-1].id) + '_' + str(res_layers[-1].id)] = (block_layers[-1].dtype, res_layers[-1].dtype, None)
                type_dict[
                    "add_" + str(block_layers[-1].id) + "_" + str(res_layers[-1].id)
                ] = (res_layers[-1].dtype, block_layers[-1].dtype, None)
            layers = list(block_layers)
        return type_dict

    def tune_and_run(
        self, mod, params, input, target, log_file, config_log=None, task_log=None
    ):
        tasks, task_weights = tvm.auto_scheduler.extract_tasks(
            mod["main"], params, target
        )
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=300, timeout=10
        )
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            # runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        prev_tuned_tasks = {}
        if task_log and os.path.exists(task_log):
            with open(task_log, "r") as yamlfile:
                logged_tasks = yaml.safe_load(yamlfile)  # Note the safe_load

            untuned_tasks = []
            untuned_tasks_weights = []

            for task, w in zip(tasks, task_weights):
                key = re.search('"(.*)"', task.workload_key).group(0).replace('"', "")
                if key not in logged_tasks["tasks"]:
                    untuned_tasks.append(task)
                    untuned_tasks_weights.append(w)
                else:
                    print("Task {} already tuned.".format(task.desc))
                    key = (
                        re.search('"(.*)"', task.workload_key).group(0).replace('"', "")
                    )
                    workload = "[" + re.search(
                        '"(?:.*)", (.*)', task.workload_key
                    ).group(1)
                    speed = logged_tasks["tasks"][key]["speed"]
                    prev_tuned_tasks[key] = {
                        "desc": str(task.desc),
                        "workload": str(workload),
                        "speed": str(speed),
                    }

            tasks = untuned_tasks
            task_weights = untuned_tasks_weights

        if tasks:
            tuner = auto_scheduler.TaskScheduler(
                tasks,
                task_weights,
                callbacks=[TuningResultsCallback(log_file=[config_log, task_log])],
            )

            if task_log:
                with open(config_log, "w") as outfile:
                    yaml.safe_dump(
                        {"graph": self.to_dict(), "tasks": prev_tuned_tasks},
                        outfile,
                        default_flow_style=False,
                    )

            tuner.tune(tune_option)

        print("Compile...")
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)

        # Create graph executor
        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))
        input_shape = input.shape
        input_dtype = input.dtype
        data_tvm = tvm.nd.array(
            (np.random.uniform(size=input_shape)).astype(input_dtype)
        )
        module.set_input("input", data_tvm)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        if config_log:
            if os.path.exists(config_log):
                with open(config_log, "r") as yamlfile:
                    cur_data = yaml.safe_load(yamlfile)  # Note the safe_load
                    if "metrics" not in cur_data:
                        cur_data["metrics"] = {}
                    cur_data["metrics"].update(
                        {"mean_inference_time": str(np.mean(prof_res))}
                    )
            else:
                cur_data = {}
                cur_data["metrics"] = {}
            with open(config_log, "w") as outfile:
                yaml.safe_dump(cur_data, outfile, default_flow_style=False)
        return np.mean(prof_res), np.std(prof_res)

    def run(self, mod, params, input, target):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)
        # Create graph executor
        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))
        input_shape = input.shape
        input_dtype = input.dtype
        data_tvm = tvm.nd.array(
            (np.random.uniform(size=input_shape)).astype(input_dtype)
        )
        module.set_input("input", data_tvm)
        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
        return np.mean(prof_res), np.std(prof_res)

    def __repr__(self) -> str:
        return "".join(["{}\n".format(b) for b in self.blocks])


ENTITY_DICT = {
    "conv1d": Conv1dEntity,
    "dense": DenseEntity,
    "pool1d": Pool1dEntity,
    "activation": ActivationEntity,
    "batch_norm": BatchNormEntity,
}

SPACE_DICT = {
    "conv1d": Conv1dSpace,
    "dense": DenseSpace,
    "pool1d": Pool1dSpace,
    "activation": ActivationSpace,
    "batch_norm": BatchNormSpace,
}

ACT_DICT = {"relu": nn.relu, "tanh": relay.tanh}

SUPPORTED_DTYPES = ["float32", "int8"]


"""From model_based_tuner.py in AutoTVM"""


def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob


def knob2point(knob, dims):
    """convert knob form (vector) to point form (single integer)"""
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k
    return p


def generate_quant_params(type_dict):
    quant_params = {}
    for k, v in type_dict.items():
        # different types, quantization necessary
        if v[0] != v[1]:
            if v[0] == "float32":
                quant_params[k] = {"output_scale": 1.0, "output_zero_point": 0}
            elif v[1] == "float32":
                quant_params[k] = {"input_scale": 1.0, "input_zero_point": 0}
            else:
                quant_params[k] = {
                    "input_scale": 1.0,
                    "input_zero_point": 0,
                    "output_scale": 1.0,
                    "output_zero_point": 0,
                }
        else:
            quant_params[k] = {}

        # check for weights
        if v[2]:
            # if weight dtype not equal to layer dtype
            if v[2] != v[1]:
                quant_params[k]["weight_scale"] = 1.0
                quant_params[k]["weight_zero_point"] = 0
    return quant_params


def generate_random_params(mod, dev=None):
    args = relay.analysis.all_vars(mod["main"])
    if not dev:
        dev = tvm.cpu(0)
    params = {}
    for arg in args:
        dt = arg.type_annotation.dtype
        params[arg.name_hint] = tvm.nd.array(
            np.random.rand(*[int(x) for x in arg.type_annotation.shape]).astype(dt),
            device=dev,
        )
    input = params.pop("input")
    return input, params


class TuningResultsCallback(TaskSchedulerCallback):
    def __init__(self, log_file):
        if log_file:
            if not isinstance(log_file, list):
                log_file = [log_file]
        else:
            log_file = []

        # for f in log_file:
        #     if os.path.exists(f):  # Remove existing log
        #         os.remove(f)

        self.log_file = log_file

    def post_tune(self, task_scheduler, task_id):
        task = task_scheduler.tasks[task_id]
        key = re.search('"(.*)"', task.workload_key).group(0).replace('"', "")
        workload = "[" + re.search('"(?:.*)", (.*)', task.workload_key).group(1)
        task_data = {
            key: {
                "workload": str(workload),
                "desc": str(task.desc),
                "speed": str(
                    task_scheduler.tasks[task_id].compute_dag.flop_ct
                    / task_scheduler.best_costs[task_id]
                    / 1e9
                ),
            }
        }
        for f in self.log_file:
            if os.path.exists(f):
                with open(f, "r") as yamlfile:
                    cur_data = yaml.safe_load(yamlfile)  # Note the safe_load
                    if "tasks" not in cur_data:
                        cur_data["tasks"] = {}
                    cur_data["tasks"].update(task_data)

            else:
                cur_data = {}
                cur_data["tasks"] = task_data

            with open(f, "w") as outfile:
                yaml.safe_dump(cur_data, outfile, default_flow_style=False)


class InvalidConfigurationError(Exception):
    def __init__(self, message) -> None:
        self.message = message
