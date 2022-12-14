from typing import Iterable
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor import TensorTuple
from hannah.nas.dataflow.dataflow_utils import find_first_op_in_dfg, find_leaf_nodes
from hannah.nas.dataflow.scoping_utils import get_id_and_update_counters, update_scope
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.parameters.parametrize import parametrize

import numpy as np


@parametrize
class DataFlowGraph(TensorExpression):
    def __init__(self, *operands, output, name: str = "dataflow") -> None:
        super().__init__(*operands, tensor_type=None, name=name)
        self.inputs = []
        self.output = output
        self.enter = []
        # reset_user(self)
        self.link_users()
        self._scopes = {}
        reset_scope_ids(self)
        self.set_scopes()

        self.collect_scopes()

    def num_nodes(self):
        n = len(self.nodes())
        return n

    def nodes(self):
        g = flatten(self)

        queue = [g]
        visited = [g]
        node_list = []
        while queue:
            current = queue.pop(-1)
            node_list.append(current.id)

            for next_node in current.next_backwards():
                if next_node not in visited:
                    queue.append(next_node)
                    visited.append(next_node)

        return node_list

    def next_backwards(self):
        return [self.output]

    def link_users(self):
        """ Link the DFG to its users and the users of the DFG to
        the DFG
        """
        def _rewire_to_placeholder(operand, node):
            """
            Parameters
            ----------
            operand : TensorExpression
                operand that we want to rewire
            node : TensorExpression
                node which uses the operand
            """
            if operand in node.operands:
                last_output = find_first_op_in_dfg(operand)
                if node in last_output.users:
                    last_output.users.remove(node)
                if node not in self.enter:
                    self.enter.append(node)
            elif set(self.operands).isdisjoint(node.operands):
                for n in node.next_backwards():
                    _rewire_to_placeholder(operand, n)
        for operand in self.operands:
            _rewire_to_placeholder(operand, self.output)
            # remove users if it is enclosed in 'self'
            for user in operand.users:
                if hasattr(self, 'enter') and user in self.enter:
                    operand.users.remove(user)
            if self not in operand.users:
                operand.users.append(self)

        self.output.users.append(self)


    def set_scope(self, current_scope, counters, visited):
        current_scope = update_scope(self, current_scope)
        scope_id = get_id_and_update_counters(current_scope, counters)
        self.id = scope_id
        queue = [*self.enter]
        visited.append(self)

        while queue:
            node = queue.pop(-1)
            node.set_scope(current_scope, counters, visited)

            leafs = []
            find_leaf_nodes(node, leafs, visited)

            while leafs:
                leaf = leafs.pop(-1)
                leaf.set_scope(current_scope, counters, visited)
                visited.append(leaf)

            for u in node.users:
                if u not in visited:
                    queue = [u] + queue
                    visited.append(u)

    def set_scopes(self):
        visited = []
        current_scope = []
        node = find_first_input(self)
        counters = {}
        queue = [node]
        visited.append(node)

        while queue:
            node = queue.pop(-1)
            node.set_scope(current_scope, counters, visited)

            leafs = []
            find_leaf_nodes(node, leafs, visited)

            while leafs:
                leaf = leafs.pop(-1)
                leaf.set_scope(current_scope, counters, visited)
                visited.append(leaf)

            for u in node.users:
                if u not in visited:
                    queue = [u] + queue
                    visited.append(u)

    def tensor_type(self):
        return self.output.tensor_type()

    def adjacency(self):
        g = flatten(self)
        g.collect_scopes()

        node_list = self.nodes()
        n = len(node_list)

        adj = np.zeros((n, n), dtype=np.int8)
        indices = {n: i for i, n in enumerate(node_list)}

        for node_id in node_list:
            node = g._scopes[node_id]

            for user in node.users:
                adj[indices[node_id]][indices[user.id]] = 1

        return adj, indices

    def __getitem__(self, key):
        return self._scopes[key]

    def __repr__(self) -> str:
        return "DataFlowGraph(id={})".format(self.id)

    def __str__(self) -> str:
        lines = []
        print_from_input(find_first_input(self), 0, [], lines)

        return_str = "\n".join(lines)
        return return_str
        # return self.__repr__()


def print_from_input(input, indent, visited, lines):
    queue = [input]
    visited.append(input)

    while queue:
        node = queue.pop(-1)

        leafs = []
        find_leaf_nodes(node, leafs, visited)
        while leafs:
            leaf = leafs.pop(-1)
            print_from_input(leaf, indent + 1, visited, lines)
            visited.append(leaf)

        lines.append('\t'*indent + f'{node.id}')
        if isinstance(node, DataFlowGraph):
            for e in node.enter:
                print_from_input(e, indent + 1, visited, lines)

        for u in node.users:
            if u not in visited:
                queue = [u] + queue
                visited.append(u)


def dataflow(func):
    def wrapper_func(*args, **kwargs):
        name = func.__name__
        operands = args
        for key, value in kwargs.items():
            if isinstance(value, int):
                kwargs[key] = DefaultInt(value)
        output = func(*args, **kwargs)

        if isinstance(output, Iterable):
            output = TensorTuple(output, name=name+".output")

        dfg = DataFlowGraph(*operands, output=output, name=name)
        return dfg

    return wrapper_func


def flatten(graph):
    delete_users(graph)
    queue = [graph]
    visited = []

    while queue:
        current = queue.pop(-1)
        visited.append(current)
        if isinstance(current, DataFlowGraph):
            if current.output not in visited:
                queue.append(current.output)

        elif isinstance(current, OpType):
            # for each operand, traverse potential DFGs and store
            # the first apperearing op
            replace_map = {}
            for i, operand in enumerate(current.operands):
                op = find_first_op_in_dfg(operand)
                replace_map[i] = op

                op.users.append(current)

                if operand not in visited:
                    queue.append(operand)

            # replace operands with non-dfg variant
            current.operands = list(current.operands)
            for idx, op in replace_map.items():
                current.operands[idx] = op
            current.operands = tuple(current.operands)

    return find_first_op_in_dfg(graph)


def unflatten(graph):
    pass


def delete_users(graph, user_to_delete=None):
    queue = [graph]
    visited = [graph]

    while queue:
        current = queue.pop(-1)
        if user_to_delete:
            if user_to_delete in current.users:
                current.users.remove(user_to_delete)
        else:
            current.users = []

        for next_node in current.next_backwards():
            if next_node not in visited:
                queue.append(next_node)
                visited.append(next_node)


def collect_users(node):
    """ Traverse graph starting from `node` and collect
    all users (including users from subsequent nodes).
    If a node_b is NOT in collect_users(node_a), this means
    that node_b is either BEFORE node_a in the graph OR it is
    in a parallel branch.

    Parameters
    ----------
    node : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    collected_users = []
    queue = [node]
    visited = []

    while queue:
        node = queue.pop(-1)
        for u in node.users:
            if u not in visited:
                queue = [u] + queue
                visited.append(u)
                collected_users.append(u)

    return collected_users


def reset_scope_ids(node):
    node.set_id(node.name)

    for next_node in node.next_backwards():
        reset_scope_ids(next_node)


def find_first_input(node):
    """Recusively traverses the graph from the given node
    back to its first input. NOTE: The traversal is via OPERANDS
    and not OUTPUT, meaning that e.g. weight Tensors that are
    included in Ops in a DFG are not returned

    Parameters
    ----------
    node : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if node.operands:
        for o in node.operands:
            return find_first_input(o)
    else:
        return node


def recursive_traversal(node : TensorExpression, hooks : list = [], hook_parameter : dict = {}, end=None):
    for hook in hooks:
        param = hook_parameter.get(hook, {})
        hook(node, **param)
    if node != end:
        for next_node in node.next_backwards():
            recursive_traversal(next_node, hooks, hook_parameter, end)
