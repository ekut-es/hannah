from hannah.nas.expressions.choice import Choice
from hannah.nas.functional_operators.op import ChoiceOp, Op, Tensor
from hannah.nas.functional_operators.operators import Add, Conv2d, Linear


def expr_product(expressions: list):
    res = None
    for expr in expressions:
        if res:
            res = res * expr
        else:
            res = expr
    return res


def expr_sum(expressions: list):
    res = None
    for expr in expressions:
        if res:
            res = res + expr
        else:
            res = expr
    return res


def expr_and(expressions):
    res = None
    for expr in expressions:
        if res:
            res = res + expr
        else:
            res = expr
    return res


ADD = 1
CHOICE = 2


class FirstAddBranch:
    pass


def which_scope(stack):
    for node in reversed(stack):
        if isinstance(node, Add):
            return ADD
        elif isinstance(node, ChoiceOp):
            return CHOICE
        elif isinstance(node, FirstAddBranch):
            return CHOICE


def extract_weights_recursive(node, visited={}, stack=[]):
    if node.id in visited:
        scope = which_scope(stack)
        if scope == ADD:
            return 0
        else:
            return visited[node.id]
    stack.append(node)
    if isinstance(node, ChoiceOp):
        exprs = []
        for o in node.options:
            w = extract_weights_recursive(o, visited, stack)
            exprs.append(w)
        c = Choice(exprs, node.switch)
        visited[node.id] = c
        stack.pop(-1)
        return c

    elif isinstance(node, Tensor) and node.id.split(".")[-1] == "weight":
        w = expr_product(node.shape())
        visited[node.id] = w
        stack.pop(-1)
        return w
    elif isinstance(node, Op):
        exprs = []
        for i, operand in enumerate(node.operands):
            if isinstance(node, Add):
                if i == 0:
                    stack.append(FirstAddBranch())
                else:
                    stack.pop(-1)
            w = extract_weights_recursive(operand, visited, stack)
            exprs.append(w)
        s = expr_sum(exprs)
        visited[node.id] = s
        stack.pop(-1)
        return s
    else:
        stack.pop(-1)
        return 0


def extract_macs_recursive(node, visited={}, stack=[]):
    if node.id in visited:
        scope = which_scope(stack)
        if scope == ADD:
            return 0
        else:
            return visited[node.id]
    stack.append(node)
    if isinstance(node, ChoiceOp):
        exprs = []
        for o in node.options:
            w = extract_macs_recursive(o, visited, stack)
            exprs.append(w)
        c = Choice(exprs, node.switch)
        visited[node.id] = c
        stack.pop(-1)
        return c
    elif isinstance(node, Op):
        if isinstance(node, Conv2d):
            output_shape = node.shape()
            volume_ofm = expr_product(output_shape)
            macs = volume_ofm * (node.in_channels / node.groups * node.kernel_size * node.kernel_size)
        elif isinstance(node, Linear):
            macs = node.in_features * node.out_features
        else:
            macs = 0
        exprs = [macs]
        for i, operand in enumerate(node.operands):
            if isinstance(node, Add):
                if i == 0:
                    stack.append(FirstAddBranch())
                else:
                    stack.pop(-1)
            w = extract_macs_recursive(operand, visited, stack)
            exprs.append(w)
        s = expr_sum(exprs)
        visited[node.id] = s
        stack.pop(-1)
        return s
    else:
        stack.pop(-1)
        return 0
