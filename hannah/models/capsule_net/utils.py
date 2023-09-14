from hannah.nas.parameters.parameters import FloatScalarParameter, IntScalarParameter, CategoricalParameter, Parameter
from omegaconf import DictConfig



def handle_parameter(mod, param, name=None):
    if isinstance(param, Parameter):
        res = param
    elif isinstance(param, DictConfig):
        assert name, "For parameter creation, name has to be specified"
        if 'min' in param:
            if isinstance(param.min, float) or isinstance(param.max, float):
                scalar_parameter = FloatScalarParameter
            else:
                scalar_parameter = IntScalarParameter
            if 'step' in param:
                res = mod.add_param(name, scalar_parameter(min=param.min, max=param.max, step_size=param.step, name=name))
            else:
                res = mod.add_param(name, scalar_parameter(min=param.min, max=param.max, name=name))
        elif 'choices' in param:
            res = mod.add_param(name, CategoricalParameter(choices=param.choices, name=name))
        else:
            raise Exception(f"Given parameter of type {type(param)} can not be handled: {param}")
    return res
