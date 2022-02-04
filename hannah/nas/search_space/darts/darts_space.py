from hannah.nas.search_space.symbolic_operator import (
    FloatRangeVector,
    SymbolicOperator,
    Choice,
    Variable,
    Constant,
    Context,
)
from hannah.nas.search_space.darts.darts_parameter_functions import (
    infer_in_channel,
    keep_channels,
    reduce_channels_by_edge_number,
    multiply_by_stem,
    reduce_and_double,
    double_channels
)
import torch
import networkx as nx
import numpy as np

from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.connectivity_constrainer import DARTSCell
from hannah.nas.search_space.modules import Add, Concat
# from hannah.nas.search_space.utils import generate_cfg_file
from hannah.nas.search_space.darts.darts_modules import (
    MixedOpWS,
    Classifier,
    Stem,
    Input,
)
from copy import deepcopy


class DARTSSpace(Space):
    def __init__(
        self, num_cells=3, reduction_cells=[1], in_edges=[4], stem_multiplier=[4]
    ):
        super().__init__()

        # search space specific configuration options
        # TODO: convert to Choice parameter
        self.cfg_options = {}
        self.cfg_options["in_edges"] = in_edges
        self.cfg_options["stem_multiplier"] = stem_multiplier
        self.weight_sharing = False

        # Define parameters
        # Variable -> parameter is inferred with a custom function
        # Constant -> directly set value
        # Choice -> choose from given options

        in_channels = Variable(
            "in_channels", func=infer_in_channel
        )  # Infer the in_channels from the input automatically
        stem_channels = Variable(
            "in_channels_stem", func=multiply_by_stem
        )  # DARTS specific channel multiplicator
        out_channels = Variable(
            "out_channels", func=keep_channels
        )  # Out_channel == in_channel
        double_out_channels = Variable(
            'out_channels', func=double_channels
        )
        out_channels_adaptive = Variable(
            "out_channels_adaptive", func=reduce_channels_by_edge_number
        )  # After the Concat, the channels are *4 -> reduce by dividing by 4
        double_out_channels_adaptive = Variable(
            "out_channels_adaptive", func=reduce_and_double
        )  # Same as above but with DARTS specific channel doubeling in red.-cells
        stride1 = Constant("stride", 1)
        stride2 = Constant("stride", 2)
        # choice = Choice("choice", 0, 1, 2, 3, 4, 5, 6, 7)
        alphas = FloatRangeVector('alphas', 0, 1, 8)

        # Define connectivity of graph
        # This could be any graph and is not really necessary, its basically just offloading the
        # graph-construction to the DARTSCell() class
        normal_cell = DARTSCell()
        normal_cell = normal_cell.add_operator_nodes()

        # The DARTSCell() class creates just the connectivity, to fill the nodes with meaningful operators
        # we create a mapping here.
        mapping = {}
        for n in normal_cell.nodes:
            if n == 0:
                # We create SymbolicOperators like this:
                # SymbolicOperators(name, module, **kwargs_of_the_respective_module)
                # If we use the name again, the new operator shares the attributes with the old one
                # Thus we can create cells that have similar topology, if desired
                mapping[n] = SymbolicOperator(
                    "input_0",
                    Input,
                    in_channels=in_channels,
                    out_channels=out_channels_adaptive,
                    stride=stride1,
                )
            elif n == 1:
                mapping[n] = SymbolicOperator(
                    "input_1",
                    Input,
                    in_channels=in_channels,
                    out_channels=out_channels_adaptive,
                    stride=stride1,
                )
            elif n in range(2, 6):
                mapping[n] = SymbolicOperator("add_{}".format(n), Add)
            elif n == 6:
                mapping[n] = SymbolicOperator("out", Concat)
            else:
                mapping[n] = SymbolicOperator(
                    "mixed_op_{}".format(n),
                    MixedOpWS,
                    alphas=alphas,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride1,
                )
                # The choice Parameter() needs an entry in the config
                # The config can be created arbitrarily (with yaml, by hand, CL-arguments, ...) but
                # it seems convinient to store the possible values here, at the creation of the Operator

        nx.relabel_nodes(normal_cell, mapping, copy=False)

        # Create another cell, this time a reduction cell
        # note the different names of the operators, meaning that a unique entry in the config
        # is required, prohibiting the sharing of Parameter between the normal- and reduction-cell
        reduction_cell = DARTSCell()
        reduction_cell = reduction_cell.add_operator_nodes()
        mapping = {}
        for n in reduction_cell.nodes:
            if n == 0:
                mapping[n] = SymbolicOperator(
                    "input_0_red",
                    Input,
                    in_channels=in_channels,
                    out_channels=double_out_channels_adaptive,
                    stride=stride1,
                )
            elif n == 1:
                mapping[n] = SymbolicOperator(
                    "input_1_red",
                    Input,
                    in_channels=in_channels,
                    out_channels=double_out_channels_adaptive,
                    stride=stride1,
                )
            elif n in range(2, 6):
                mapping[n] = SymbolicOperator("add_{}_red".format(n), Add)
            elif n == 6:
                mapping[n] = SymbolicOperator("out_red", Concat)
            elif isinstance(n, tuple) and n[0] in [0, 1]:
                mapping[n] = SymbolicOperator(
                    "mixed_op_{}_red".format(n),
                    MixedOpWS,
                    alphas=alphas,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride2,
                )
            else:
                mapping[n] = SymbolicOperator(
                    "mixed_op_{}_red".format(n),
                    MixedOpWS,
                    alphas=alphas,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride1,
                )

        nx.relabel_nodes(reduction_cell, mapping, copy=False)

        out_idx = 6
        input_0_idx = 0
        input_1_idx = 1

        # DARTS preprocessing stem
        stem = SymbolicOperator("stem0", Stem, C_out=stem_channels)

        # create the desired num of cells from the prototypes and
        # edit parameter depending on the reduction cells
        cells = [deepcopy(normal_cell) for i in range(num_cells)]
        for idx in reduction_cells:
            cells[idx] = deepcopy(reduction_cell)
            if idx < len(cells) - 1:
                # if the previous cell was a reduction cell, we must modify
                # the incoming data from the skip connection
                list(cells[idx + 1].nodes)[input_0_idx].params["stride"] = stride2
                list(cells[idx + 1].nodes)[input_0_idx].params[
                    "out_channels"
                ] = double_out_channels_adaptive

        # The data coming from the stem is not concatenated, therefore we can leave the channels as is
        list(cells[0].nodes)[input_0_idx].params["out_channels"] = double_out_channels if 0 in reduction_cells else out_channels
        list(cells[0].nodes)[input_1_idx].params["out_channels"] = double_out_channels if 0 in reduction_cells else out_channels
        list(cells[1].nodes)[input_0_idx].params["out_channels"] = double_out_channels if 1 in reduction_cells else out_channels

        # Add cell nodes and edges to SearchSpace (self) and
        for i in range(len(cells)):
            self.add_nodes_from([n for n in cells[i].nodes])
            self.add_edges_from([e for e in cells[i].edges])

        # connect stem to cell-nodes
        self.add_edge(stem, list(cells[0].nodes)[input_0_idx])
        self.add_edge(stem, list(cells[0].nodes)[input_1_idx])
        self.add_edge(stem, list(cells[1].nodes)[input_0_idx])

        # connect cells + skip-connections
        for i in range(len(cells)):
            if i < len(cells) - 2:
                self.add_edge(
                    list(cells[i].nodes)[out_idx], list(cells[i + 2].nodes)[input_0_idx]
                )
            if i < len(cells) - 1:
                self.add_edge(
                    list(cells[i].nodes)[out_idx], list(cells[i + 1].nodes)[input_1_idx]
                )

        # create and add post-process (i.e. fully connected) to graph
        post = SymbolicOperator(
            "post", Classifier, C=in_channels, num_classes=Choice("classes", 10)
        )
        self.add_node(post)
        self.add_edge(list(cells[-1].nodes)[out_idx], post)

    def prepare_weight_sharing(self, size):
        cfg_dims = self.get_config_dims()
        random_cfg = self.get_random_cfg(cfg_dims)
        ctx = Context(random_cfg)
        input = torch.ones(size)
        instance, _ = self.infer_parameters(input, ctx)
        self.instance = instance
        self.weight_sharing = True

    def get_ctx(self):
        return self.ctx

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
                        cfg[k][k_] = np.random.uniform(v_['min'], v_['max'], v_['size'])
                        np.random.uniform()
            else:
                cfg[k] = int(np.random.choice(v))
        return cfg

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


def get_space_and_instance(cfg):
    space = DARTSSpace(
        num_cells=12,
        reduction_cells=[
            i for i in range(num_cells) if i in [num_cells // 3, 2 * num_cells // 3]
        ],
    )
    ctx = Context(config=cfg)
    input = torch.ones([1, 3, 32, 32])
    instance, out1 = space.infer_parameters(input, ctx)
    return instance


if __name__ == "__main__":
    num_cells = 8
    reduction_cells = [
        i for i in range(num_cells) if i in [num_cells // 3, 2 * num_cells // 3]
    ]
    space = DARTSSpace(num_cells=num_cells, reduction_cells=reduction_cells)
    cfg_dims = space.get_config_dims()
    file_name = "./hannah/nas/search_space/examples/darts/cfg_dims.yml"
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
