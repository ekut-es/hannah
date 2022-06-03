
def find_first_op_in_dfg(node):
    if hasattr(node, 'output'):
        return find_first_op_in_dfg(node.output)
    else:
        return node
