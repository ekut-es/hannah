import tune
import yaml
from space import NetworkSpace, point2knob, NetworkEntity
import numpy as np
from hannah_tvm import config
from omegaconf import OmegaConf


board_config_path = "/home/moritz/Dokumente/Hiwi/code/hannah-tvm/hannah_tvm/conf/board/jetsontx2_cpu.yaml"

# with open(board_config_path) as f:
#     board_config = yaml.safe_load(f)

board = OmegaConf.load(board_config_path)

cfg_space = NetworkSpace()
cfg_space.from_yaml(
    "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator/configs/test_net.yaml"
)

idx = np.random.choice(np.prod(cfg_space.dims()) - 1)
print("{}|{}".format(idx, np.prod(cfg_space.collapsed_dims())))
cfg = point2knob(idx, cfg_space.collapsed_dims())
print("CFG:", cfg)
input_shape = (1, 40, 101)

try:
    net = NetworkEntity(cfg_space, cfg_space.expand_config(cfg))
    tune.tune(net, input_shape, board)
except Exception as e:
    print(str(e))
