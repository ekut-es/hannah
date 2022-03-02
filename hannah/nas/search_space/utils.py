import numpy as np
import torch


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def get_random_cfg(cfg_dims):
    """ Create random config
    Returns
    -------
    dict
        a random config
    """
    cfg = {}
    for k, v in cfg_dims.items():
        if isinstance(v, dict):
            cfg[k] = {}
            for k_, v_ in v.items():
                if isinstance(v_, list):
                    cfg[k][k_] = int(np.random.choice(v_))
                elif isinstance(v_, dict) and 'min' in v_ and 'size':
                    # cfg[k][k_] = np.random.uniform(v_['min'], v_['max'], v_['size'])
                    cfg[k][k_] = torch.FloatTensor(v_['size']).uniform_(v_['min'], v_['max'])
                elif isinstance(v_, dict):
                    key = np.random.choice(list(v_.keys()))
                    cfg[k][k_] = key
                    cfg[key] = {}
                    for k__, v__ in v_[key].items():
                        cfg[key][k__] = int(np.random.choice(v__))

        else:
            cfg[k] = int(np.random.choice(v))
    return cfg
