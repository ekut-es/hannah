_OPS = {}
_SHAPE_FUNCS = {}
_CONVERSIONS = {}


def op(name, *operands, **attributes):
    return _OPS[name].create_op(*operands, **attributes)


def shape(op_name):
    if op_name not in _SHAPE_FUNCS:
        # if no shape func is registered for the op, just pass-through the tensor type
        # of the first operand (which we assume to be the input)
        op_name = 'Default'
    return _SHAPE_FUNCS[op_name]
