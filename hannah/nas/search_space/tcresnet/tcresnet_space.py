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
from hannah.nas.search_space.utils import get_first_cfg


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
    cfg_dims = space.get_config_dims()
    # cfg = get_random_cfg(cfg_dims)
    cfg = get_first_cfg(cfg_dims)
    # cfg = {k: {k_: 0 if k_ == 'stride' else v_ for k_, v_ in v.items()} for k, v in cfg.items()}
    ctx = Context(cfg)
    x = torch.ones([1, 40, 101])
    instance, out = space.infer_parameters(x, ctx, verbose=True)
    print(instance)
    print(out.shape)

    # keys = list(flatten_config(cfg).keys())
    # df = pd.DataFrame(columns=keys + ['valid'])

    # for i in range(500000):
    #     if i % 1000 == 0:
    #         print('{}|{}'.format(i, 500000))
    #     cfg = get_random_cfg(cfg_dims)
    #     flat_data = flatten_config(cfg)
    #     try:
    #         ctx = Context(cfg)
    #         x = torch.ones([1, 40, 101])
    #         instance, out = space.infer_parameters(x, ctx, verbose=False)
    #         valid = 1
    #     except Exception as e:
    #         valid = 0
    #     flat_data['valid'] = valid
    #     df = df.append(flat_data, ignore_index=True)
    # if i % 10000 == 0:
    #     df.to_csv('/home/moritz/projects/hannah/hannah/nas/search_space/tcresnet/data.csv')
    # df.to_csv('/home/moritz/projects/hannah/hannah/nas/search_space/tcresnet/data.csv')


if __name__ == '__main__':
    main()
