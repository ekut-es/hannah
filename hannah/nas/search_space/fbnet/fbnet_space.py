import torch
# import torch.nn as nn
import numpy as np
from hannah.nas.search_space.symbolic_operator import SymbolicOperator, Constant, FloatRangeVector, Context
from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.fbnet.fbnet_modules import ConvBNRelu, MixedOperation, CLF, PRIMITIVES


class FBNetSpace(Space):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # TODO: Make configurable
        OPERATIONS = ["ir_k3_e1", "ir_k3_s2", "ir_k3_e3",
                      "ir_k3_e6", "ir_k5_e1", "ir_k5_s2",
                      "ir_k5_e3", "ir_k5_e6", "skip"]

        INPUT_SHAPES = [16,
                        16,   24,  24,  24,
                        24,   32,  32,  32,
                        32,   64,  64,  64,
                        64,   112, 112, 112,
                        112, 184, 184, 184,
                        184]

        CHANNEL_SIZES = [16,
                         24,  24,  24,  24,
                         32,  32,  32,  32,
                         64,  64,  64,  64,
                         112, 112, 112, 112,
                         184, 184, 184, 184,
                         352]

        STRIDES = [1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   2, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   1]
        num_layers = len(STRIDES)

        first = SymbolicOperator('first', ConvBNRelu, input_depth=3, output_depth=16, kernel=3, stride=2,
                                 pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")

        nodes = []
        for layer_id in range(num_layers):
            op = SymbolicOperator('MixedOp_{}'.format(layer_id),
                                  MixedOperation,
                                  C_in=Constant('c_in', INPUT_SHAPES[layer_id]),
                                  C_out=Constant('c_out', CHANNEL_SIZES[layer_id]),
                                  stride=Constant('stride', STRIDES[layer_id]),
                                  proposed_operations=Constant('ops', {op: PRIMITIVES[op] for op in OPERATIONS}),
                                  alphas=FloatRangeVector('alphas', 0, 1, len(OPERATIONS)))
            nodes.append(op)
            self.add_node(op)
            if layer_id > 0:
                self.add_edge(nodes[layer_id-1], op)
        self.add_edge(first, nodes[0])

        clf = SymbolicOperator('clf', CLF, in_channel=CHANNEL_SIZES[-1])
        self.add_edge(nodes[-1], clf)

    def get_config_dims(self):
        # cfg options of the nodes
        cfg = super().get_config_dims()

        # search space specific options
        for k, v in self.cfg_options.items():
            # cfg.update({k: list(range(len(v)))})
            # NOTE currently, op parameter are idxes and global SS parameter have to
            # be the final values
            cfg.update({k: v})
        return cfg

    # TODO: Simplify and maybe move to other symbolic_space.py
    def get_random_cfg(self, cfg_dims):
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

                        np.random.uniform()
            else:
                cfg[k] = int(np.random.choice(v))
        return cfg

    def prepare_weight_sharing(self, size):
        cfg_dims = self.get_config_dims()
        random_cfg = self.get_random_cfg(cfg_dims)
        ctx = Context(random_cfg)
        input = torch.ones(size)
        instance, _ = self.infer_parameters(input, ctx)
        self.instance = instance
        self.weight_sharing = True

    def sample(self, size):
        cfg_dims = self.get_config_dims()
        random_cfg = self.get_random_cfg(cfg_dims)
        ctx = self.get_ctx()
        ctx.set_cfg(random_cfg)
        input = torch.ones(size)
        if self.weight_sharing:
            instance = self.instance
            alphas = {}
            for node in self.nodes:
                if 'alphas' in node.params:
                    alphas[ctx.relabel_dict[node]] = node.params['alphas'].get(node, ctx)
            for node in instance.nodes:
                if node in alphas:
                    node.set_alphas(alphas[node])
        else:
            instance, _ = self.infer_parameters(input, ctx)

        return instance

    def get_ctx(self):
        return self.ctx


if __name__ == "__main__":
    num_cells = 8
    reduction_cells = [
        i for i in range(num_cells) if i in [num_cells // 3, 2 * num_cells // 3]
    ]
    space = FBNetSpace(None)
    cfg_dims = space.get_config_dims()
    # if not os.path.isfile(file_name):
    #     generate_cfg_file(cfg_dims, file_name)
    cfg = space.get_random_cfg(cfg_dims)
    # generate_cfg_file(
    #     cfg, "./hannah/nas/search_space/examples/darts/random_darts_model.yaml"
    # )
    ctx = Context(config=cfg)
    input = torch.ones([1, 3, 32, 32])
    instance, out1 = space.infer_parameters(input, ctx)
    print(out1.shape)

    mse = torch.nn.MSELoss()
    output = instance(input)
    loss = mse(output, torch.ones(output.shape))
    loss.backward()
    print(loss)
