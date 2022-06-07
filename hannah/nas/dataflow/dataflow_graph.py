from typing import Iterable
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorTuple
from hannah.nas.dataflow.dataflow_utils import find_first_op_in_dfg, find_leaf_nodes
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor import Tensor


class DataFlowGraph(TensorExpression):
    def __init__(self, *operands, output, name: str = "dataflow") -> None:
        super().__init__(*operands, tensor_type=None, name=name)
        self.inputs = []
        self.operand_to_input_map = {}
        if self.operands:
            for i, o in enumerate(self.operands):
                inp = Tensor(name='input')
                self.inputs.append(inp)
                self.operand_to_input_map[o] = inp

        self.output = output
        self.link_users()
        first_inp = find_first_input(self)
        set_scope_ids(first_inp, [], [], {})

    def link_users(self):
        def _rewire_to_placeholder(operand, node, placeholder):
            if operand in node.operands:
                last_output = find_first_op_in_dfg(operand)
                last_output.users.remove(node)

                placeholder.users.append(node)
            elif isinstance(node, DataFlowGraph):
                _rewire_to_placeholder(operand, node.output, placeholder)
            elif isinstance(node, OpType):
                for o in node.operands:
                    _rewire_to_placeholder(operand, o, placeholder)

        for operand, inp in self.operand_to_input_map.items():
            last_output = find_first_op_in_dfg(operand)
            last_output.users.append(self)
            self.users.append(inp)
            _rewire_to_placeholder(operand, self.output, inp)

    def __repr__(self) -> str:
        return "DataFlowGraph(id={})".format(self.id)


def dataflow(func):
    def wrapper_func(*args, **kwargs):
        name = func.__name__
        operands = args
        output = func(*args, **kwargs)

        if isinstance(output, Iterable):
            output = TensorTuple(output, name=name+".output")

        dfg = DataFlowGraph(*operands, output=output, name=name)
        return dfg

    return wrapper_func


# FIXME: I'd rather have these methods in a different place but
# one has to be careful to avoid circular imports. This works for now.
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


def reset_scope_ids(node):
    node.set_id(node.name)
    for o in node.operands:
        reset_scope_ids(o)


def find_first_input(node):
    if node.operands:
        for o in node.operands:
            return find_first_input(o)
    else:
        return node
