from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.test.network import residual_block
from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.dataflow.transformations.graph_tranformer import GraphTransformer

from hannah.nas.dataflow.dataflow_graph import DataFlowGraph, dataflow, find_first_input, flatten


def get_dict(graph):
    queue = [graph]
    visited = []
    hierarchy_dict = {}

    while queue:
        current = queue.pop(-1)
        visited.append(current)
        name = current.id.split('.')
        sub_dict = hierarchy_dict

        for i, key in enumerate(name):
            try:
                k = int(key)
            except Exception:
                k = key
            k = key

            if k not in sub_dict:
                sub_dict[k] = {}
            if i != len(name) - 1:
                sub_dict = sub_dict[k]
            else:
                sub_dict[k]['texpr'] = current

        for operand in current.operands:
            queue.append(operand)
    return hierarchy_dict


def get_parent_scope(node):
    return node.id.split('.')[:-2]


def check_dfg_change(node, successor):
    node_parent_scope = ".".join(get_parent_scope(node))
    successor_parent_scope = ".".join(get_parent_scope(successor))
    return node_parent_scope != successor_parent_scope


def get_dfg_depth(node):
    scope = get_parent_scope(node)
    return int(len(scope) / 2)


def create_dfgs(op_graph):
    queue = [find_first_input(op_graph)]
    enter = None
    output = None
    previous = None
    operands = []

    while queue:
        current = queue.pop(-1)
        parent_scope = get_parent_scope(current)

        if previous:
            if check_dfg_change(previous, current):
                if get_dfg_depth(previous) > 0:
                    dfg = DataFlowGraph(output=previous, name=get_parent_scope(previous)[-2])
                    dfg.enter = enter
                enter = current
                print(f'Enter dfg {get_parent_scope(current)}')
                if get_dfg_depth(previous) != get_dfg_depth(current):
                    print(f"changed dfg depth from {get_dfg_depth(previous)} to {get_dfg_depth(current)}")

        for user in current.users:
            queue.append(user)

        previous = current




def make_dataflow_graph(hierarchy_dict, dfgs, output_dict):
    for key in hierarchy_dict.keys():
        for num in hierarchy_dict[key].keys():
            if 'texpr' in hierarchy_dict[key][num]:
                output = hierarchy_dict[key][num]['texpr']

                name_list = output.id.split('.')
                if len(name_list) > 2:
                    name = output.id.split('.')[-4]
                else:
                    dfgs.append(output)
                    return output
            else:
                output = make_dataflow_graph(hierarchy_dict[key][num], dfgs, output_dict)
                name = key
            dfg = DataFlowGraph(output=output, name=f'{name}')
            output_dict[output]= dfg
            dfgs.append(dfg)
    return dfgs[0]


def unflatten(graph):
    hierarchy = {}

    queue = [graph]
    visited = []

    while queue:
        current = queue.pop(-1)
        visited.append(current)
        name = current.id.split('.')


def write_down(graph):
    print(graph.id)
    for operand in graph.operands:
        write_down(operand)


@dataflow
def exchange_block(input):
    input_tensor = input.tensor_type()
    weight = weight_tensor(shape=(DefaultInt(1), input_tensor['c'], DefaultInt(1), DefaultInt(1)), name='weight')
    c = op("Conv2d", input, weight, stride=DefaultInt(1))
    relu = OpType(c, name='Relu')
    return relu


def test_graph_transformer():
    # Create a network and flatten the graph
    input = batched_image_tensor(shape=(1, 3, 32, 32), name='input')
    graph = residual_block(input, stride=IntScalarParameter(1, 2), output_channel=IntScalarParameter(4, 512, 4))



    # flat = flatten(graph)
    # create_dfgs(flat)


    # d = get_dict(flat)
    # dfgs = []
    # output_dict = {}
    # make_dataflow_graph(d, dfgs, output_dict)
    # dfg = unflatten(flat)

    transformer = GraphTransformer(graph)

    def transform(source, target):
        args = [op for op in source.operands]
        kwargs = {}

        return args, kwargs

    transformer.transform('conv_relu', exchange_block, transform)
    print()


if __name__ == '__main__':
    test_graph_transformer()
