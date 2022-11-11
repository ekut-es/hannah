import numpy as np
import torch
import itertools


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, (int, np.int64)), 'kernel size should be either `int` or `tuple`'
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


def get_first_cfg(cfg_dims):
    cfg = {}
    for k, v in cfg_dims.items():
        if isinstance(v, dict):
            cfg[k] = {}
            for k_, v_ in v.items():
                if isinstance(v_, list):
                    cfg[k][k_] = v_[0]
                elif isinstance(v_, dict) and 'min' in v_ and 'size':
                    # TODO: Change
                    raise NotImplementedError()
                    # cfg[k][k_] = np.random.uniform(v_['min'], v_['max'], v_['size'])
                    cfg[k][k_] = torch.FloatTensor(v_['size']).uniform_(v_['min'], v_['max'])
                elif isinstance(v_, dict):
                    key = list(v_.keys())[0]
                    cfg[k][k_] = key
                    cfg[key] = {}
                    for k__, v__ in v_[key].items():
                        cfg[key][k__] = v__[0]

        else:
            cfg[k] = int(np.random.choice(v))
    return cfg


def flatten_config(config):
    flatten_cfg = {}
    for k, v in config.items():
        for k_, v_ in v.items():
            flatten_cfg[k + '.' + k_] = v_
    return flatten_cfg


def unflatten_config(config):
    nested_cfg = {}
    for k, v in config.items():
        modkey, parkey = k.split('.')
        if modkey not in nested_cfg:
            nested_cfg[modkey] = {}
        nested_cfg[modkey][parkey] = v
    return nested_cfg


def generate_config_permutations(config_dims):
    keys, values = zip(*config_dims.items())
    for v in itertools.product(*values):
        yield unflatten_config(dict(zip(keys, v)))
