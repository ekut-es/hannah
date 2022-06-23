_OPS = {}
_SHAPE_FUNCS = {}
_CONVERSIONS = {}


def op(name, *operands, **attributes):
    return _OPS[name].create_op(*operands, **attributes)


def shape(op_name):
    return _SHAPE_FUNCS[op_name]
