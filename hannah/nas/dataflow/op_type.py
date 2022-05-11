from hannah.nas.parameters.parametrize import parametrize
from typing import Any


class OpType:
    def __init__(self, name, *operands, **attributes):
        self.name = name
        self.operands = list(operands)
        self.attributes = attributes

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for i, arg in enumerate(args):
            self.operands[i] = arg
        for key, value in kwds.items():
            self.attributes[key] = value

        return self


    # TODO:
    def __repr__(self) -> str:
        return self.name + '(\n' +  \
                ' '.join(['{}, '.format(o) for o in self.operands]) +\
                ' '.join(['{}: {}'.format(key, val) for key, val in self.attributes.items()]) + ')\n'
