from copy import deepcopy
import inspect
from ..core.parametrized import is_parametrized


def _create_parametrize_wrapper(parameters, cls):
    parameter_list = list(parameters.values())
    old_init_fn = cls.__init__

    def init_fn(self, *args, **kwargs):
        self._PARAMETERS = {}
        self._annotations = {}

        for num, arg in enumerate(args):
            if is_parametrized(arg):
                name = parameter_list[num + 1].name
                self._PARAMETERS[name] = arg
                self._annotations[name] = parameter_list[num + 1]._annotation
        for name, arg in kwargs.items():
            if is_parametrized(arg):
                self._PARAMETERS[name] = arg
                self._annotations[name] = parameters[name]._annotation

        # TODO:
        cls.sample = sample
        cls.set_current = set_current
        cls.instantiate = instantiate
        cls.check = check
        cls.set_params = set_params
        self._parametrized = True
        old_init_fn(self, *args, **kwargs)

    return init_fn


def parametrize(cls=None):
    def parametrize_function(cls):
        init_fn = cls.__init__
        init_sig = inspect.signature(init_fn)

        new_init_fn = _create_parametrize_wrapper(init_sig.parameters, cls)
        cls.__init__ = new_init_fn

        return cls

    if cls:
        return parametrize_function(cls)

    return parametrize_function


def sample(self):
    for _key, param in self._PARAMETERS.items():
        param.sample()


def set_current(self, value):
    self.set_params(**value)


def set_params(self, **kwargs):
    for key, value in kwargs.items():
        assert key in self._PARAMETERS, "{} has no parameter {}".format(self, key)

        if not isinstance(value, dict) and key in self._annotations and not isinstance(value, self._annotations[key]):
            raise TypeError('Value must be of type {} but is {}'.format(self._annotations[key], type(value)))
        if is_parametrized(value):
            self._PARAMETERS[key] = value
            setattr(self, key, value)  # TODO: Do we want this to work?
        else:
            self._PARAMETERS[key].set_current(value)


# required for Protocol
def check(self, value):
    # TODO:
    pass


def instantiate(self):
    instance = deepcopy(self)
    instance._parametrized = False

    for key, param in instance._PARAMETERS.items():
        instantiated_value = param.instantiate()
        instance._PARAMETERS[key] = instantiated_value
        setattr(instance, key, instantiated_value)
    return instance
