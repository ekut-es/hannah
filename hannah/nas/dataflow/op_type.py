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
        ret += "{}(".format(self.name) + \
               "".join(["\t{}, \n".format(o) for o in self.operands]) + \
               "".join(["\t{}={}".format(key, str(attr)) for key, attr in self.attributes.items()]) + \
               ")"
        return ret



