import inspect
from ..core.parametrized import is_parametrized


def _create_parametrize_wrapper(parameters, cls):
    parameter_list = list(parameters.values())
    old_init_fn = cls.__init__

    def init_fn(self, *args, **kwargs):
        self._PARAMETERS = {}

        for num, arg in enumerate(args):
            if is_parametrized(arg):
                # breakpoint()
                name = parameter_list[num + 1].name
                self._PARAMETERS[name] = arg
        for name, arg in kwargs.items():
            if is_parametrized(arg):
                self._PARAMETERS[name] = arg

        # TODO:
        cls.sample = sample
        cls.set_params = set_params
        # cls.instantiate = instantiate
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


def set_params(self, **kwargs):
    for key, value in kwargs.items():
        assert key in self._PARAMETERS, "{} has no parameter {}".format(self, key)
        self._PARAMETERS[key].set_current(value)


def instantiate(self):
    self._parametrized = False
