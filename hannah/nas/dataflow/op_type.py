from hannah.nas.parameters.parametrize import parametrize
from typing import Any


class OpType:
    def __init__(self, name, *operands, **attributes):
        self.name = name
        self.id = name
        self.operands = list(operands)
        self.attributes = attributes

    def output_tensor(self):
        # TODO: what happens with multiple operands?
        # => here one has to define how the op
        # changes the tensor
        return self.operands[0].output_tensor()

    def __repr__(self) -> str:
        ret = ""
        ret += "%{} = " + \
               "{}(".format(self.name) + \
               "".join(["%{}, " for _ in range(len(self.operands))]) + \
               "".join(["{}={}".format(key, str(attr)) for key, attr in self.attributes.items()]) + \
               ")"
        for operand in reversed(self.operands):
            ret = repr(operand) + '\n' + ret
        return ret



