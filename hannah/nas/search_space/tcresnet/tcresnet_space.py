from hannah.nas.search_space.pruner import Pruner
from hannah.nas.search_space.symbolic_constraint_solver import SymbolicConstrainer
from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.symbolic_operator import Context, Choice, Variable, infer_in_channel, infer_padding_symbolic
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import hannah.conf  # noqa
from hannah.modules.config_utils import get_model
import torch
import numpy as np
from hannah.nas.search_space.torch_converter import TorchConverter
from hannah.nas.search_space.utils import get_random_cfg


class TCResNetSpace(Space):
    def __init__(self, config, parameterization=None):
        super().__init__()
        model_config = instantiate(config.model)
        self.model = get_model(model_config)
        # self.model = TCResNetModel(config=config)
        converter = TorchConverter()
        converter.convert_model_to_space(self.model, self)

        if parameterization:
            for symop in self.nodes:
                attrs = {}
                if 'kernel_size' in symop.params:
                    attrs['kernel_size'] = Choice('kernel_size', 1, 3, 5)
                if 'padding' in symop.params:
                    attrs['padding'] = Variable('padding', func=infer_padding_symbolic)
                if 'stride' in symop.params:
                    attrs['stride'] = Choice('stride', 1, 2)
                if 'dilation' in symop.params:
                    attrs['dilation'] = Choice('dilation', 1, 3, 9)
                if 'in_channels' in symop.params:
                    attrs['in_channels'] = Variable('in_channels', func=infer_in_channel)
                if 'out_channels' in symop.params:
                    attrs['out_channels'] = Choice('out_channels', *range(4, 512, 4))
                if 'num_features' in symop.params:  # and "BatchNorm" in str(symop.target_cls):
                    attrs['num_features'] = Variable('num_features', func=infer_in_channel)
                if 'in_features' in symop.params:  # and "BatchNorm" in str(symop.target_cls):
                    attrs['in_features'] = Variable('in_features', func=infer_in_channel)

                symop.update_parameters(**attrs)


def test_basic_model(config):
    space = TCResNetSpace(config)
    ctx = Context(None)
    x = torch.ones([1, 40, 101])
    space.model.eval()
    out_model = space.model(x)
    instance, _ = space.infer_parameters(x, ctx)
    state_dict = {}
    for key, value in space.model.state_dict().items():
        count = key.count(".") - 1
        state_dict['nodes.' + key.replace(".", '_', count)] = value
    instance.load_state_dict(state_dict)
    instance.eval()
    out_instance = instance(x)
    np.testing.assert_allclose(out_model.detach(), out_instance.detach())


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):
    # test_basic_model(config)

    space = TCResNetSpace(config, parameterization=True)
    x = torch.ones([1, 40, 101])

    pruner = Pruner(space)
    channel_constrainer = SymbolicConstrainer(space)
    cfg_dims = space.get_config_dims()
    cfg = get_random_cfg(cfg_dims)
    cfg = channel_constrainer.constrain_output_channels(cfg)
    cfg = pruner.find_next_valid_config(x, cfg, exclude_keys=['out_channels', 'kernel_size', 'dilation'])
    ctx = Context(cfg)
    instance, out = space.infer_parameters(x, ctx, verbose=True)

    print(instance)
    print(out.shape)


if __name__ == '__main__':
    main()
