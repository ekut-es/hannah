import speech_recognition
from speech_recognition.models import factory
import yaml
import space
import numpy as np

from space import NetworkSpace, Conv1dSpace, BlockSpace, DenseSpace, Pool1dSpace
from space import Conv1dEntity, BlockEntity, NetworkEntity, DenseEntity, Pool1dEntity

from tvm import relay
from tvm.relay import nn
import tvm

layerk1_c24 = Conv1dSpace(
    kernel_size=[1],
    dtype=["int"],
    bw=[4, 6, 8],
    act=[nn.relu],
    strides=[2],
    batch_norm=[True],
    fold_batch_norm=[True],
    channels=[24],
)

layerk1_c32 = Conv1dSpace(
    kernel_size=[1],
    dtype=["int"],
    bw=[4, 6, 8],
    act=[nn.relu],
    strides=[2],
    batch_norm=[True],
    channels=[32],
)

layerk1_c48 = Conv1dSpace(
    kernel_size=[1],
    dtype=["int"],
    bw=[4, 6, 8],
    act=[nn.relu],
    strides=[2],
    batch_norm=[True],
    channels=[48],
)

layerk3 = Conv1dSpace(
    kernel_size=[3], dtype=["int"], bw=[4, 6, 8], act=[nn.relu], channels=[16]
)

layerk9_s2_c24 = Conv1dSpace(
    kernel_size=[9],
    dtype=["int"],
    bw=[4, 6, 8],
    act=[nn.relu],
    strides=[2],
    repeat=[1],
    padding=[4],
    batch_norm=[True],
    channels=[24],
)

layerk9_s1_c24 = Conv1dSpace(
    kernel_size=[9],
    dtype=["int"],
    bw=[4, 6, 8],
    strides=[1],
    repeat=[1],
    padding=[4],
    batch_norm=[True],
    channels=[24],
)

layerk9_s2_c32 = Conv1dSpace(
    kernel_size=[9],
    dtype=["int"],
    bw=[4, 6, 8],
    act=[nn.relu],
    strides=[2],
    repeat=[1],
    padding=[4],
    batch_norm=[True],
    channels=[32],
)

layerk9_s1_c32 = Conv1dSpace(
    kernel_size=[9],
    dtype=["int"],
    bw=[4, 6, 8],
    strides=[1],
    repeat=[1],
    padding=[4],
    batch_norm=[True],
    channels=[32],
)

layerk9_s2_c48 = Conv1dSpace(
    kernel_size=[9],
    dtype=["float"],
    bw=[32],  # bw=[4,6,8],
    act=[nn.relu],
    strides=[2],
    repeat=[1],
    padding=[4],
    batch_norm=[True],
    channels=[48],
)

layerk9_s1_c48 = Conv1dSpace(
    kernel_size=[9],
    dtype=["float"],
    bw=[32],  # bw=[4,6,8],
    strides=[1],
    repeat=[1],
    padding=[4],
    batch_norm=[True],
    channels=[48],
)


layer_pool = Pool1dSpace(pool_size=[int(13)], strides=[int(13)])
layer_fc = DenseSpace(labels=[12], dtype=["int"], bw=[8])


# group layers in blocks
start_block = BlockSpace(layers=[layerk3])

# If one block consists of the same layer multiple times, one can just repeat it
basic_block1 = BlockSpace(
    layers=[layerk9_s2_c24, layerk9_s1_c24],
    residual=BlockSpace(layers=[layerk1_c24]),
    repeat=[1],
    block_act=[nn.relu],
)

basic_block2 = BlockSpace(
    layers=[layerk9_s2_c32, layerk9_s1_c32],
    residual=BlockSpace(layers=[layerk1_c32]),
    repeat=[1],
    block_act=[nn.relu],
)

basic_block3 = BlockSpace(
    layers=[layerk9_s2_c48, layerk9_s1_c48],
    residual=BlockSpace(layers=[layerk1_c48]),
    repeat=[1],
    block_act=[nn.relu],
)

dense_block = BlockSpace(layers=[layer_pool, layer_fc])
# dense_block = BlockSpace(layers=[layer_fc])
# Add blocks to network
cfg_space = NetworkSpace()
cfg_space.add_block(start_block)
cfg_space.add_block(basic_block1)
cfg_space.add_block(basic_block2)
cfg_space.add_block(basic_block3)
cfg_space.add_block(dense_block)


idx = 59048
cfg = space.point2knob(idx, cfg_space.collapsed_dims())
print(np.prod(cfg_space.dims()))
print(cfg)
net = NetworkEntity(cfg_space, cfg_space.expand_config(cfg))

inp = relay.var("input", shape=(1, 40, 101))
mod = net.to_relay(inp)
# mod = relay.frontend.common.infer_shape(inp, mod)
print(mod)

args = relay.analysis.all_vars(mod["main"])

params = {}
for arg in args:
    dt = arg.type_annotation.dtype
    print(arg.name_hint, " ", dt)
    params[arg.name_hint] = tvm.nd.array(
        np.random.rand(*[int(x) for x in arg.type_annotation.shape]).astype(dt)
    )
input = params.pop("input")


ex = tvm.relay.create_executor("graph")

program = ex.evaluate(mod["main"])
out = program(input, **params)
print("Out", out)
target = "llvm"
tasks = tvm.auto_scheduler.extract_tasks(mod, params, target)[0]
print(tasks)
