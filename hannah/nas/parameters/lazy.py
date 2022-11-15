from hannah.nas.parameters.parameters import Parameter
from copy import deepcopy


class Lazy:
    def __init__(self, klass) -> None:
        self.klass = klass
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        return deepcopy(self)

    def sample(self):
        return self.klass(*[arg.evaluate() if hasattr(arg, 'evaluate') else arg for arg in self.args], **{name: arg.evaluate() if hasattr(arg, 'evaluate') else arg for name, arg in self.kwargs.items()})
