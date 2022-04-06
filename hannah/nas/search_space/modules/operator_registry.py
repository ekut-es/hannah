_OPERATORS = {}


def register_op(op_name=None):
    def register(klass):
        _OPERATORS[op_name] = klass
        return klass

    return register
