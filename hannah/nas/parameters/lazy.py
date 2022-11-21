from hannah.nas.expressions.shapes import identity_shape
from copy import deepcopy


class Lazy:
    def __init__(self, klass, shape_func=identity_shape) -> None:
        self.klass = klass
        self.args = ()
        self.kwargs = {}
        self.shape_func = shape_func

    def __call__(self, id: str, *args, inputs: list = [], **kwargs):
        self._id = id
        self.inputs = inputs

        self.args = args
        self.kwargs = kwargs

        return deepcopy(self)

    @property
    def id(self):
        if hasattr(self, '_id'):
            return self._id
        else:
            raise AttributeError("Lazy objects have to be initialized to have an id.")

    @property
    def shape(self):
        if hasattr(self, '_output_shape'):
            return self._output_shape
        else:
            self._output_shape = self.shape_func(*[inp.shape if hasattr(inp, 'shape') else inp for inp in self.inputs], kwargs=self.kwargs)
            return self._output_shape

    def __repr__(self) -> str:
        if hasattr(self, '_id'):
            name = self._id
        else:
            name = self.klass.__name__
        return "Lazy({})".format(name)

    def instantiate(self):
        return self.klass(*[arg.evaluate() if hasattr(arg, 'evaluate') else arg for arg in self.args], **{name: arg.evaluate() if hasattr(arg, 'evaluate') else arg for name, arg in self.kwargs.items()})
