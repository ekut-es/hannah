import hannah
from hannah.models import factory
import yaml
import space
import numpy as np

from space import (
    NetworkSpace,
    Conv1dSpace,
    BlockSpace,
    DenseSpace,
    Pool1dSpace,
    ActivationSpace,
    BatchNormSpace,
)
from space import (
    Conv1dEntity,
    BlockEntity,
    NetworkEntity,
    DenseEntity,
    Pool1dEntity,
    ActivationEntity,
    BatchNormEntity,
)

from tvm import auto_scheduler
from tvm.contrib import graph_executor

from tvm import relay
from tvm.relay import nn
import tvm
import os

cfg_space = NetworkSpace()
cfg_space.from_yaml(
    "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator/configs/test_net_2.yaml"
)

idx = np.random.choice(np.prod(cfg_space.dims()) - 1)
print("{}|{}".format(idx, np.prod(cfg_space.collapsed_dims())))
cfg = space.point2knob(idx, cfg_space.collapsed_dims())
print("CFG:", cfg)

# cfg = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
# idx = space.knob2point(cfg, cfg_space.collapsed_dims())

try:
    net = NetworkEntity(cfg_space, cfg_space.expand_config(cfg))
    print(net)
    inp = relay.var("input", shape=(1, 40, 101))
    quant_seq = net.quantization_sequence(inp)
    quant_params = space.generate_quant_params(quant_seq)

    kwargs = {"quant_params": quant_params}

    mod = net.to_relay(inp, **kwargs)
    # mod = relay.frontend.common.infer_shape(inp, mod)
    print(mod)

    input, params = space.generate_random_params(mod)

    # target = tvm.target.Target("cuda")
    target = tvm.target.Target("llvm")

    tasks, task_weights = tvm.auto_scheduler.extract_tasks(mod["main"], params, target)

    network = "test"
    dtype = "float32"
    batch_size = 1
    input_shape = input.shape
    # log_file = "%s-B%d-%s.json" % (network, batch_size, target.kind.name)
    log_file = (
        "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator/data/testing/log.json"
    )
    conf_log = "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator/data/testing/logs/{}.yaml".format(
        idx
    )
    task_log = "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator/data/testing/logs/tasks.yaml"

    def run_tuning():
        print("Begin tuning...")
        # if os.path.exists(log_file):  # Remove existing log
        #     os.remove(log_file)
        # if os.path.exists(conf_log):
        #     os.remove(conf_log)
        cost_mean, cost_std = net.tune_and_run(
            mod, params, input, target, log_file, config_log=conf_log, task_log=task_log
        )
        print("Cost (mean|std): {}|{}".format(cost_mean, cost_std))
        return cost_mean

    run_tuning()
except Exception as e:
    print(str(e))
