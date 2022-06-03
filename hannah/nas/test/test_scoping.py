from hannah.nas.dataflow.dataflow_graph import dataflow, DataFlowGraph
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import UndefinedInt
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from hannah.nas.ops import axis, tensor, batched_image_tensor, float_t


@dataflow
def conv_relu(input: TensorType,
              output_channel=IntScalarParameter(4, 64),
              kernel_size=CategoricalParameter([1, 3, 5]),
              stride=CategoricalParameter([1, 2])):

    weight = tensor((axis('o', size=output_channel),
                     axis('i', size=UndefinedInt()),
                     axis('kh', size=kernel_size),
                     axis('kw', size=kernel_size)),
                    dtype=float_t(),
                    name='weight')

    op = OpType(input, weight, stride=stride, name='conv2d')
    relu = OpType(op, name='relu')
    return relu


@dataflow
def block(input: TensorType,
          expansion=FloatScalarParameter(1, 6),
          output_channel=IntScalarParameter(4, 64),
          kernel_size=CategoricalParameter([1, 3, 5]),
          stride=CategoricalParameter([1, 2])):

    out = conv_relu(input,
                    output_channel=output_channel*expansion,
                    kernel_size=kernel_size,
                    stride=stride)
    out = conv_relu(out,
                    output_channel=output_channel,
                    kernel_size=1,
                    stride=1)
    return out


def update_scope(node, current_scope):
    to_remove = []
    for scope in current_scope:
        if isinstance(scope, Tensor):
            to_remove.append(scope)
        elif isinstance(scope, OpType) and node in scope.users:
            to_remove.append(scope)
        elif isinstance(scope, DataFlowGraph) and scope in node.operands:
            to_remove.append(scope)

    new_scope = []
    for s in current_scope:
        if s not in to_remove:
            new_scope.append(s)
        else:
            # if a scope is removed, all lower-hierarchy scopes
            # are removed too because we assume strictly nested scopes
            # i.e. not overlapping
            break
    new_scope.append(node)
    return new_scope


def get_id_and_update_counters(current_scope, counters):
    if len(current_scope) > 1:
        scope = '.'.join([current_scope[-2].id, current_scope[-1].name])
    else:
        scope = current_scope[-1].name
    if scope not in counters:
        counters[scope] = 0
    else:
        counters[scope] += 1

    return '{}.{}'.format(scope, counters[scope])


def set_scope_ids(node, visited, current_scope, counters):
    current_scope = update_scope(node, current_scope)
    scope_id = get_id_and_update_counters(current_scope, counters)
    node.set_id(scope_id)
    leafs = []
    visited.append(node)
    find_leaf_nodes(node, leafs, visited)
    for leaf in leafs:
        set_scope_ids(leaf, visited, current_scope, counters)
    for u in node.users:
        if u not in visited:
            set_scope_ids(u, visited, current_scope, counters)


def find_leaf_nodes(node, leafs, visited):
    for o in node.operands:
        if o not in visited:
            if isinstance(o, Tensor):
                leafs.append(o)
            else:
                find_leaf_nodes(o, leafs, visited)


def traverse_users(node, visited):
    print(node.id)
    leafs = []
    visited.append(node)
    find_leaf_nodes(node, leafs, visited)
    for leaf in leafs:
        traverse_users(leaf, visited)
    for u in node.users:
        if u not in visited:
            traverse_users(u, visited)


def test_repeat():
    input = batched_image_tensor(name='input')
    graph = block(input)
    graph = block(graph)
    set_scope_ids(input, [], [], {})

    traverse_users(input, [])
    print()

    assert isinstance(graph, DataFlowGraph)


if __name__ == '__main__':
    test_repeat()
