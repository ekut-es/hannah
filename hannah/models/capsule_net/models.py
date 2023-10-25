from omegaconf import OmegaConf
import torch
import torch.nn as nn
from hannah.nas.expressions.logic import And
from hannah.nas.expressions.metrics import conv2d_macs, conv2d_weights
from hannah.nas.expressions.shapes import conv2d_shape
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.parameters.iterators import RangeIterator
from hannah.nas.parameters.parameters import IntScalarParameter, CategoricalParameter
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.choice import Choice
from hannah.nas.expressions.types import Int

from hannah.models.capsule_net.utils import handle_parameter
from hannah.models.capsule_net.expressions import depth_aware_downsampling, depth_aware_sum, expr_sum, num_layer_constraint
from hannah.models.capsule_net.operators import (
    Convolution,
    DepthwiseConvolution,
    Linear,
    PointwiseConvolution,
    Relu,
    Identity,
    BatchNorm,
    Pooling,
)


conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)

@parametrize
class Activation(nn.Module):
    def __init__(self, params, id) -> None:
        super().__init__()
        self.params = params
        self.id = id

        # FIXME: Introduce choice here
        relu_act = Relu(self.id)
        identity_act = Identity(self.id)

        self.mods = nn.ModuleDict({"relu": relu_act, "identity": identity_act})
        # self.mods = {"relu": relu_act, "identity": identity_act}
        self.choice = handle_parameter(self, params, "choice")
        self.active_module = Choice(self.mods, self.choice)

    def initialize(self):
        self.active_module.evaluate().initialize()

    def forward(self, x):
        return self.active_module.evaluate()(x)


@parametrize
class ReducedComplexitySpatialOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()


@parametrize
class ExpandReduce(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.params = params
        self.inputs = inputs
        input_shape = inputs[0]
        self.id = id
        self.in_channels = input_shape[1]

        # FIXME: Share parameters over patterns
        self.out_channels = handle_parameter(self, self.params.convolution.out_channels, "out_channels")
        self.expand_ratio = handle_parameter(self, self.params.expand_reduce.ratio, "expand_ratio")

        self.expanded_channels = Int(self.expand_ratio * self.in_channels)
        self.expansion = PointwiseConvolution(self.expanded_channels, self.id + '.expand', inputs)
        self.bn0 = BatchNorm(None, self.id, [self.expansion.shape])
        self.act0 = Activation(params.activation, self.id + ".activation0")

        # FIXME: Replace with ReducedComplexitySpatialOperator
        self.spatial_correlations = DepthwiseConvolution(params.convolution, self.id, [self.expansion.shape])
        self.bn1 = BatchNorm(None, self.id, [self.spatial_correlations.shape])
        self.act1 = Activation(params.activation, self.id + ".activation1")
        self.reduction = PointwiseConvolution(self.out_channels, self.id + '.reduce', [self.spatial_correlations.shape])

    def initialize(self):
        self.expansion.initialize()
        self.bn0.initialize()
        self.act0.initialize()
        self.spatial_correlations.initialize()
        self.bn1.initialize()
        self.act1.initialize()
        self.reduction.initialize()

    def forward(self, x):
        out = self.expansion(x)
        out = self.bn0(out)
        out = self.act0(out)
        out = self.spatial_correlations(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.reduction(out)
        return out

    @property
    def shape(self):
        return self.reduction.shape

    @property
    def macs(self):
        return self.expansion.macs + self.spatial_correlations.macs + self.reduction.macs

    @property
    def weights(self):
        return self.expansion.weights + self.spatial_correlations.weights + self.reduction.weights


@parametrize
class ReduceExpand(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.params = params
        self.id = id
        self.inputs = inputs
        self.input_shape = inputs[0]

        self.in_channels = self.input_shape[1]

        # FIXME: Share parameters over patterns
        self.out_channels = handle_parameter(self, self.params.convolution.out_channels, f"out_channels")
        self.reduce_ratio = handle_parameter(self, self.params.reduce_expand.ratio, f"reduce_ratio")

        self.reduced_channels = Int(self.reduce_ratio * self.in_channels)
        self.reduction = PointwiseConvolution(self.reduced_channels, self.id + '.expand', inputs)
        self.bn0 = BatchNorm(None, self.id, [self.reduction.shape])
        self.act0 = Activation(params.activation, self.id + ".activation0")

        # FIXME: convolution out channels?
        self.conv = Convolution(params.convolution, self.id, [self.reduction.shape])
        self.bn1 = BatchNorm(None, self.id, [self.conv.shape])
        self.act1 = Activation(params.activation, self.id + ".activation1")
        self.expansion = PointwiseConvolution(self.out_channels, self.id + '.expand', [self.conv.shape])

    def initialize(self):
        self.reduction.initialize()
        self.bn0.initialize()
        self.act0.initialize()
        self.conv.initialize()
        self.bn1.initialize()
        self.act1.initialize()
        self.expansion.initialize()

    def forward(self, x):
        out = self.reduction(x)
        out = self.bn0(out)
        out = self.act0(out)
        out = self.conv(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.expansion(out)
        return out

    @property
    def shape(self):
        return self.expansion.shape

    @property
    def macs(self):
        return self.reduction.macs + self.conv.macs + self.expansion.macs

    @property
    def weights(self):
        return self.reduction.weights + self.conv.weights + self.expansion.weights


@parametrize
class Residual(nn.Module):
    def __init__(self, params, input_shape, output_shape, id) -> None:
        super().__init__()
        self.id = id
        self.params = params
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.in_channels = input_shape[1]
        self.out_channels = output_shape[1]
        self.in_fmap = input_shape[2]
        self.out_fmap = output_shape[2]
        self.stride = Int(Ceil(self.in_fmap / self.out_fmap))

        # FIXME: Add potential alternative downsampling (pooling)
        # FIXME: Move lazy conv to operators
        self.conv = conv2d(self.id + ".conv",
                           inputs=[input_shape],
                           in_channels=self.in_channels,
                           out_channels=self.out_channels,
                           kernel_size=1,
                           stride=self.stride,
                           padding=0)
        self.bn = BatchNorm(None, self.id, [self.conv.shape])
        self.activation = Activation(params.activation, self.id + ".activation")

    def initialize(self):
        self.downsample_conv = self.conv.instantiate()
        self.bn.initialize()
        self.activation.initialize()

    def forward(self, x):
        out = self.downsample_conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


    @property
    def shape(self):
        return self.conv.shape

    @property
    def macs(self):
        return conv2d_macs(self.input_shape, self.shape, self.conv.kwargs)

    @property
    def weights(self):
        return conv2d_weights(self.input_shape, self.shape, self.conv.kwargs)



@parametrize
class Pattern(nn.Module):
    def __init__(self, params, input_shape, id) -> None:
        super().__init__()
        self.params = params
        self.input_shape = input_shape
        self.id = id

        # shared parameters
        self.stride = handle_parameter(self, params.stride, "stride")
        self.out_channels = handle_parameter(self, params.out_channels, "out_channels")

        conv_params = OmegaConf.create({'convolution': {'stride': self.stride, 'out_channels': self.out_channels},
                                        'pooling': {'stride': self.stride}}, flags={"allow_objects": True})
        self.params = OmegaConf.merge(self.params, conv_params)

        # FIXME: Check whether the self.add_params is necessary (probably yes, for child parameters to be registered when calling search_space.parameterization())
        convolution = self.add_param(f"{self.id}.convolution", Convolution(self.params.convolution, f"{self.id}.convolution", [self.input_shape]))
        expand_reduce = self.add_param(f"{self.id}.expand_reduce", ExpandReduce(self.params, f"{self.id}.expand_reduce", [self.input_shape]))
        reduce_expand = self.add_param(f"{self.id}.reduce_expand", ReduceExpand(self.params, f"{self.id}.reduce_expand", [self.input_shape]))
        pooling = self.add_param(f"{self.id}.pooling", Pooling(self.params.pooling, f"{self.id}.pooling", [self.input_shape]))

        self.mods = nn.ModuleDict({'convolution': convolution,
                                   'expand_reduce': expand_reduce,
                                   'reduce_expand': reduce_expand,
                                   'pooling': pooling})

        self.choice = self.add_param(f"{self.id}.choice", CategoricalParameter(choices=params.choices))
        self.active_module = Choice(self.mods, self.choice)
        self.bn = BatchNorm(None, self.id, [self.shape])
        # FIXME: Make activation dependent on last activation in active module
        self.activation = Activation(params.activation, self.id + ".activation")

    def initialize(self):
        self.active_module.evaluate().initialize()  # FIXME: See whether this works or whether we have to initialize all modules
        self.activation.initialize()

    def forward(self, x):
        out =  self.active_module.evaluate()(x)
        out = self.activation(out)
        return out

    @property
    def shape(self):
        return[self.active_module.get('shape')[0],
               self.active_module.get('shape')[1],
               self.active_module.get('shape')[2],
               self.active_module.get('shape')[3]]

    @property
    def macs(self):
        return self.active_module.get('macs')

    @property
    def weights(self):
        return self.active_module.get('weights')



@parametrize
class Block(nn.Module):
    def __init__(self, params, input_shape, id) -> None:
        super().__init__()
        self.id = id
        self.input_shape = input_shape
        self.depth = handle_parameter(self, params.depth, 'depth')
        self.mods = nn.ModuleList()
        self.params = params

        next_input = self.input_shape
        for i in RangeIterator(self.depth, instance=False):
            mod = self.add_param(f"{self.id}.pattern.{i}", Pattern(params.patterns, next_input, f"{self.id}.pattern.{i}"))
            self.mods.append(mod)
            next_input = mod.shape

        self.last_mod = Choice(self.mods, self.depth - 1)
        self.residual = self.add_param(f"{self.id}.residual", Residual(params=params.residual,
                                                                       input_shape=input_shape,
                                                                       output_shape=self.shape,
                                                                       id=f"{self.id}.residual"))

    def initialize(self):
        for d in RangeIterator(self.depth, instance=False):
            self.mods[d].initialize()
        self.residual.initialize()

    def forward(self, x):
        out = x
        for d in RangeIterator(self.depth, instance=True):
            out = self.mods[d](out)
        res_out = self.residual(x)
        out = torch.add(out, res_out)
        return out

    @property
    def shape(self):
        # shape = self.last_mod.get('shape')
        shape = [self.last_mod.get("shape")[0],
                 self.last_mod.get("shape")[1],
                 self.last_mod.get("shape")[2],
                 self.last_mod.get("shape")[3]]
        return shape # FIXME: validate whether this works

    @property
    def macs(self):
        mac_list = []
        for d in RangeIterator(self.depth, instance=False):
            mac_list.append(self.mods[d].macs)

        return depth_aware_sum(mac_list, self.depth) + self.residual.macs

    @property
    def weights(self):
        weight_list = []
        for d in RangeIterator(self.depth, instance=False):
            weight_list.append(self.mods[d].weights)

        return depth_aware_sum(weight_list, self.depth)  + self.residual.weights


@parametrize
class Stem(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.params = params
        self.id = id
        self.inputs = inputs
        self.input_shape = self.inputs[0]

        self.conv = self.add_param(f"{self.id}.convolution", Convolution(params.convolution, self.id, self.inputs))
        self.bn = BatchNorm(None, self.id, [self.conv.shape])
        self.act = Activation(params.activation, self.id)

    def initialize(self):
        self.conv.initialize()
        self.bn.initialize()
        self.act.initialize()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

    @property
    def shape(self):
        return self.conv.shape

    @property
    def macs(self):
        return self.conv.macs

    @property
    def weights(self):
        return self.conv.weights


@parametrize
class Neck(nn.Module):
    def __init__(self, params, input_shape) -> None:
        super().__init__()
        self.params = params
        self.input_shape = input_shape

    @property
    def shape(self):
        pass # FIXME:

    @property
    def macs(self):
        pass

    @property
    def weights(self):
        pass


@parametrize
class SearchSpace(nn.Module):
    def __init__(self, name, params, input_shape, labels) -> None:
        super().__init__()
        self.name = name
        self.input_shape = input_shape
        self.labels = labels
        self.params = params
        self.num_blocks = IntScalarParameter(self.params.num_blocks.min, self.params.num_blocks.max)
        self.add_param("num_blocks", self.num_blocks)

        self.blocks = nn.ModuleList()
        # FIXME: Subsampling
        self.stem = self.add_param("stem", Stem(params.stem, "stem", [self.input_shape]))
        next_input = self.stem.shape

        for n in RangeIterator(self.num_blocks, instance=False):
            block = self.add_param(f"block.{n}", Block(params.block, next_input, f"block.{n}"))
            self.blocks.append(block)

            next_input = block.shape

        last_block = Choice(self.blocks, self.num_blocks - 1)

        # self.neck = self.add_param("neck", Neck(params.neck, next_input))
        # NOTE: One can apparently not just write last_block.get("shape") and then index on the receiving end
        self.head = self.add_param("head", Linear(params.head,
                                                [last_block.get("shape")[0],
                                                 last_block.get("shape")[1],
                                                 last_block.get("shape")[2],
                                                 last_block.get("shape")[3]],
                                                self.labels))


        strides = []
        block_depths = []
        for n, p in self.parametrization(flatten=True).items():
            if 'stride' in n:
                strides.append(p)
            elif 'depth' in n:
                block_depths.append(p)
        self.downsampling = depth_aware_downsampling(strides, self.parametrization(flatten=True))

        self.cond(And(self.downsampling <= self.input_shape[2], self.downsampling >= (self.input_shape[2] / self.params.min_reduction)))
        block_depths.sort(key=lambda x: x.id)
        num_blocks = self.parametrization(flatten=True)['num_blocks']
        self.num_layers = num_layer_constraint(block_depths, num_blocks)
        # self.cond(self.num_layers <= self.params.max_layer)
        # self.cond(self.macs <= 1000000000)
        print()

    def initialize(self):
        self.stem.initialize()
        for n in RangeIterator(self.num_blocks, instance=False):
            self.blocks[n].initialize()
        # self.neck.initialize()
        self.head.initialize()

    def forward(self, x):
        out = self.stem(x)
        for n in RangeIterator(self.num_blocks, instance=True):
            out = self.blocks[n](out)
        # out = self.neck(out)
        out = self.head(out)
        return out

    @property
    def shape(self):
        # FIXME: Doesnt work right now
        return self.head.shape

    @property
    def macs(self):
        mac_list = []
        for d in RangeIterator(self.num_blocks, instance=False):
            mac_list.append(self.blocks[d].macs)

        blocks = depth_aware_sum(mac_list, self.num_blocks)
        return self.stem.macs + blocks + self.head.macs

    @property
    def weights(self):
        weight_list = []
        for d in RangeIterator(self.num_blocks, instance=False):
            weight_list.append(self.blocks[d].weights)

        blocks = depth_aware_sum(weight_list, self.num_blocks)
        return self.stem.weights + blocks + self.head.weights

