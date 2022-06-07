
from hannah.nas.dataflow.tensor import Tensor


def find_first_op_in_dfg(node):
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
