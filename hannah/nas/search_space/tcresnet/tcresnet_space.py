from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.symbolic_operator import Context
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import hannah.conf  # noqa
from hannah.modules.config_utils import get_model
import torch.nn as nn
import torch
import numpy as np
from hannah.nas.search_space.torch_converter import TorchConverter


class MethodWrapper(nn.Module):
    def __init__(self, method, args, kwargs) -> None:
        super().__init__()
        # self.name = name
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        xargs = []
        if isinstance(x, list):
            x, *xargs = x
        return getattr(x, self.method)(*xargs, *self.args, **self.kwargs)


class FunctionWrapper(nn.Module):
    def __init__(self, function, args=[], kwargs={}) -> None:
        super().__init__()
        # self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        if isinstance(x, list):
            return self.function(*x, *self.args, **self.kwargs)
        else:
            return self.function(x, *self.args, **self.kwargs)

    def __repr__(self):
        return "Wrapped({})".format(self.function)


class TCResNetSpace(Space):
    def __init__(self, config):
        super().__init__()
        model_config = instantiate(config.model)
        self.model = get_model(model_config)
        # self.model = TCResNetModel(config=config)
        converter = TorchConverter()
        converter.convert_model_to_space(self.model, self)


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):
    space = TCResNetSpace(config)
    ctx = Context(None)
    x = torch.ones([1, 40, 101])
    space.model.eval()
    out_model = space.model(x)
    print("Out Model", out_model)
    instance, _ = space.infer_parameters(x, ctx)
    state_dict = {}
    for key, value in space.model.state_dict().items():
        count = key.count(".") - 1
        state_dict['nodes.' + key.replace(".", '_', count)] = value
    instance.load_state_dict(state_dict)
    instance.eval()
    out_instance = instance(x)
    print("Out Instance", out_instance)
    np.testing.assert_allclose(out_model.detach(), out_instance.detach())


if __name__ == '__main__':
    main()
