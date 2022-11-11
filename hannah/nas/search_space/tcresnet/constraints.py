from constraint import *
from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
import hydra
from omegaconf import DictConfig
from hannah.nas.search_space.symbolic_operator import Choice
import numpy as np


def equal_input_channel_constraint(*args):
    if len(args) == 0:
        return False
    else:
        return args.count(args[0]) == len(args)


def find_next_channel(space, node):
    if 'out_channels' in node.params:
        return '{}_{}'.format(node.name, 'out_channels')
    else:
        in_edges = space.in_edges(node)
        if len(in_edges) == 0:
            raise Exception("No channels found")
        elif len(in_edges) > 1:
            raise Exception("Found fork")
        elif len(in_edges) == 1:
            for u, v in in_edges:
                return find_next_channel(space, u)


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):
    problem = Problem()
    space = TCResNetSpace(config, parameterization=True)
    for node in space:
        print(node.name)
        for key, par in node.params.items():
            if isinstance(par, Choice):
                if "downsample" in node.name:
                    problem.addVariable('{}_{}'.format(node.name, key), par.values)
                else:
                    problem.addVariable('{}_{}'.format(node.name, key), [np.random.choice(par.values)])
        if 'add' in node.name:
            in_edges = space.in_edges(node)
            channel_names = []
            for u, v in in_edges:
                channel_names.append(find_next_channel(space, u))
            problem.addConstraint(equal_input_channel_constraint, channel_names)

    for s in problem.getSolutionIter():
        print(s)
    # sol = problem.getSolution()



if __name__ == '__main__':
    main()