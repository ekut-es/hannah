
from typing import Optional
from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.expressions.placeholder import DefaultInt, UndefinedInt


# FIXME: Rename to "last" op
def find_first_op_in_dfg(node):
    if hasattr(node, 'output'):
        return find_first_op_in_dfg(node.output)
    else:
        return node


def find_next_dataflow(node):
    if hasattr(node, 'output'):
        return node
    else:
        assert len(node.users) < 2, "Next DataflowGraph is ambiguous"
        return find_next_dataflow(node.users[0])


def remove_old_users(node):
    if hasattr(node, 'output'):
        return find_first_op_in_dfg(node.output)
    else:
        return node


def find_leaf_nodes(node, leafs, visited):
    for o in node.operands:
        if o not in visited:
            if isinstance(o, Tensor):
                leafs.append(o)
            else:
                leafs.append(o)
                find_leaf_nodes(o, leafs, visited)


def traverse_by_users(node):
    def _traverse_by_users(node, visited):
        print(node.id)
        leafs = []
        visited.append(node)
        find_leaf_nodes(node, leafs, visited)
        for leaf in leafs:
            _traverse_by_users(leaf, visited)
        for u in node.users:
            if u not in visited:
                _traverse_by_users(u, visited)
    _traverse_by_users(node, [])


def process_int(x: Optional[int]):
    if isinstance(x, int):
        return DefaultInt(x)
    elif isinstance(x, Expression):
        return x
    elif x is None:
        return UndefinedInt()
