import torch
from hannah.nas.search_space.modules.complex_operators import MBInvertedConvLayer
from hannah.nas.search_space.modules.primitive_operators import Conv2dAct, DepthwiseSeparableConvolution, ReLU6
from hannah.nas.search_space.symbolic_operator import Choice, OneOf, SymbolicOperator, Context, Variable, infer_in_channel
from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.proxyless.proxyless_modules import MobileInvertedResidualBlock, Classifier
from hannah.nas.search_space.proxyless.proxyless_parameter import restricted_stride
from hannah.nas.search_space.utils import get_random_cfg


# TODO: Rename
# not really the Proxyless space, just inspired by it
class ProxylessSpace(Space):
    def __init__(self,
                 n_cell=20,
                 width_mult=1, n_classes=10):
        super().__init__()
        relu = ReLU6()
        stride = Choice('stride', 1, 2, func=restricted_stride)
        width = Choice('width', *[i for i in range(4, 128, 4)])

        # first conv
        first_conv = SymbolicOperator("FirstConv",
                                      Conv2dAct,
                                      in_channels=3,
                                      out_channels=width,
                                      kernel_size=3,
                                      stride=stride,
                                      act_func=relu)

        mb_conv_proto = SymbolicOperator("MBConv",
                                         MBInvertedConvLayer,
                                         in_channels=Variable("in_channels", func=infer_in_channel),
                                         out_channels=width,
                                         kernel_size=Choice('kernel_size', 3, 5, 7),
                                         stride=stride,
                                         expand_ratio=Choice('expand_ratio', 3, 6))

        dw_conv_proto = SymbolicOperator("DWConv",
                                         DepthwiseSeparableConvolution,
                                         in_channels=Variable("in_channels", func=infer_in_channel),
                                         out_channels=width,
                                         kernel_size=Choice('kernel_size', 3, 5, 7),
                                         stride=stride)

        blocks = []
        for i in range(n_cell):
            shortcut = None
            mb_conv = mb_conv_proto.new(mb_conv_proto.name + '_{}'.format(i))
            dw_conv = dw_conv_proto.new(dw_conv_proto.name + '_{}'.format(i))

            conv = SymbolicOperator('BlockChoice_{}'.format(i),
                                    MobileInvertedResidualBlock,
                                    conv=OneOf('oneof', mb_conv, dw_conv),
                                    shortcut=shortcut)
            blocks.append(conv)

        classifier = SymbolicOperator('Classifier',
                                      Classifier,
                                      in_channels=Variable("in_channels", func=infer_in_channel),
                                      last_channels=Variable("in_channels", func=infer_in_channel),
                                      n_classes=n_classes)
        self.add_edge(first_conv, blocks[0])
        for i in range(len(blocks) - 1):
            self.add_edge(blocks[i], blocks[i+1])
        self.add_edge(blocks[-1], classifier)

    def prepare(self, input_size):
        self.input_size = input_size

    def sample(self, config):
        ctx = self.get_ctx()
        ctx.set_cfg(config)
        input = torch.ones(self.input_size)
        instance, _ = self.infer_parameters(input, ctx)
        return instance

    def get_ctx(self, cfg=None):
        if not hasattr(self, 'ctx'):
            if not cfg:
                cfg_dims = self.get_config_dims()
                cfg = get_random_cfg(cfg_dims)
            self.ctx = Context(config=cfg)
            self.ctx.max_reductions = 2
        else:
            if cfg:
                self.ctx.config = cfg
        return self.ctx

    def get_supernetwork(self, x):
        cfg_dims = self.get_config_dims()
        cfg = self.get_random_cfg(cfg_dims)
        ctx = Context(config=cfg)
        instance, _ = self.infer_parameters(x, ctx)
        return instance


if __name__ == "__main__":
    space = ProxylessSpace()
    cfg_dims = space.get_config_dims()
    cfg = space.get_random_cfg(cfg_dims)
    print(cfg)
    ctx = Context(config=cfg)
    input = torch.ones([2, 3, 32, 32])
    supernetwork = space.get_supernetwork(input)
    instance, out1 = space.infer_parameters(input, ctx)
    print(out1.shape)

    mse = torch.nn.MSELoss()
    output = instance(input)
    loss = mse(output, torch.ones(output.shape))
    loss.backward()
    print(loss)
