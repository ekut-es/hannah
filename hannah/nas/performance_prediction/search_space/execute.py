import tvm
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
import search_space.space as space

from hannah_tvm import measure

import yaml
from search_space.space import NetworkSpace, point2knob, NetworkEntity
import numpy as np
from hannah_tvm import config
from omegaconf import OmegaConf
from tvm import autotvm
import os
import shutil
from hannah_tvm.compiler import Compiler_Ext, get_compiler_options
import tvm.micro as micro
from tvm.contrib import utils
import time
import traceback
from tvm import rpc
from tvm.rpc.tracker import Tracker
from automate.config import AutomateConfig
from automate.context import AutomateContext
from tvm import auto_scheduler


def execute_and_measure(net, input_shape, board, measure_context, log_file):
    inp = relay.var("input", shape=input_shape)
    quant_seq = net.quantization_sequence(inp)
    quant_params = space.generate_quant_params(quant_seq)
    kwargs = {"quant_params": quant_params}
    try:
        mod = net.to_relay(inp, **kwargs)
    except Exception as e:
        prof_res = [100000]
        print("Relay conversion failed")
        print(str(e))
        return prof_res
    input, params = space.generate_random_params(mod)

    print("Converted net to relay ...")

    print("Instantiated measure_context ...")

    target = tvm.target.Target(board.target)
    target_host = tvm.target.Target(board.target_host)

    print(target)
    print(target_host)

    if target.kind == "cuda":
        if "arch" in target.attrs:
            autotvm.measure.measure_methods.set_cuda_target_arch(target.attrs["arch"])
        else:
            logger.warning("CUDA target has no architecture attribute")

    build_cfg = {}
    if target.kind == "c":
        build_cfg = {"tir.disable_vectorize": True}
    try:
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config=build_cfg):
                lib = relay.build(mod, target=target, params=params)
        print("Build lib ...")
        if board.local:
            dev = tvm.device(str(target), 0)
            module = graph_executor.GraphModule(lib["default"](dev))

        else:
            temp = utils.tempdir()
            path = temp.relpath("lib.tar")
            lib.export_library(path)
            tracker = measure_context.tracker
            tracker_sess = rpc.connect_tracker(tracker.host, tracker.port)
            remote = tracker_sess.request(board.name, priority=0, session_timeout=60)
            remote.upload(path)
            dev = remote.cpu()
            rlib = remote.load_module("lib.tar")
            dev = remote.cpu()

            module = graph_executor.GraphModule(rlib["default"](dev))

        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
        module.set_input("input", data_tvm)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
    except Exception as e:
        prof_res = [100000]
        print("Execution failed")
        print(str(e))

    return prof_res


if __name__ == "__main__":
    board_config_path = "/home/moritz/Dokumente/Hiwi/code/hannah-tvm/hannah_tvm/conf/board/jetsontx2_cpu.yaml"

    # with open(board_config_path) as f:
    #     board_config = yaml.safe_load(f)

    board = OmegaConf.load(board_config_path)

    cfg_space = NetworkSpace()
    cfg_space.from_yaml(
        "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator/configs/test_net.yaml"
    )

    idx = 10000  # np.random.choice(np.prod(cfg_space.dims()) - 1)
    print("{}|{}".format(idx, np.prod(cfg_space.collapsed_dims())))
    cfg = point2knob(idx, cfg_space.collapsed_dims())
    print("CFG:", cfg)
    input_shape = (1, 40, 101)

    try:
        net = NetworkEntity(cfg_space, cfg_space.expand_config(cfg))
        execute_and_measure(net, input_shape, board)
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
