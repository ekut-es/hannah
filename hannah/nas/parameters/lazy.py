from hannah.nas.parameters.parameters import Parameter
from hannah.nas.expressions.shapes import identity_shape
from copy import deepcopy


class Lazy:
    def __init__(self, klass, shape_func=identity_shape) -> None:
        self.klass = klass
        self.args = ()
        self.kwargs = {}
        self.shape_func = shape_func

    def __call__(self, id: str, *args, **kwargs):
        self._id = id
        self.args = args
        self.kwargs = kwargs

        return deepcopy(self)

    @property
    def id(self):
        if hasattr(self, '_id'):
            return self._id
        else:
            raise AttributeError("Lazy objects have to be initialized to have an id.")

    def shape(self, input_shape):
        return self.shape_func(input_shape, self.kwargs)

    def __repr__(self) -> str:
        return "Lazy({})".format(self.klass.__name__)

    def sample(self):
        return self.klass(*[arg.evaluate() if hasattr(arg, 'evaluate') else arg for arg in self.args], **{name: arg.evaluate() if hasattr(arg, 'evaluate') else arg for name, arg in self.kwargs.items()})
