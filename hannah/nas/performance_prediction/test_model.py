from .search_space.space import NetworkSpace, NetworkEntity, point2knob

cfg_space = NetworkSpace()
cfg_space.from_dict(
    [
        {
            "block": [
                {
                    "conv1d": [
                        {
                            "kernel_size": [1, 3, 5, 7, 9],
                            "dtype": ["int8", "float32"],
                            "channels": [x for x in range(1, 128)],
                        }
                    ]
                }
            ]
        }
    ]
)

print(cfg_space)


for idx in range(0, 10):
    print(idx)
    cfg = point2knob(idx, cfg_space.collapsed_dims())
    print(cfg)
