!!! warning
    The search spaces in HANNAH are currently under construction. If you run into bugs, please contact us.


# Search Spaces 

Search spaces in HANNAH are directed graphs (DAGs) where the nodes are **Ops** or **Tensors** and the edges indicate data movement.

!!! note
    Search spaces are not executable themselves but need an [Executor](#executor) which uses the current parametrization state to
    build a `forward`.



```python
from hannah.nas.functional_operators.operators import Conv2d
```

![Graph illustration](../../assets/graph.jpg)

## Basic Building Blocks

### Search Space Wrapper
To define the beginning and end of a search space, the definition has to be enclosed in a function returning the (last node of the) search space graph. 
This function must use the `@search_space` decorator to indicate that this is the main search space enclosing function. 



```python
from hannah.nas.functional_operators.op import search_space
```

### Ops & Tensors

**Op** nodes represent the operators used in the networks of the search space. Their basic syntax is

```python
# usage
var_name = Operator(param0=val0, param1=val1, ...)(*operands)
```

**Tensor** nodes indicate the presence of data in the graph. They do not themselves contain actual values when 
the search space graph is defined (the actual data is managed by the [Executor](#executor)). The tensor node 
defines attributes that the data has at this point in the graph (e.g., shape, axis names, datatypes, ...). 


```python
from hannah.nas.functional_operators.operators import Conv2d
from hannah.nas.functional_operators.op import Tensor

@search_space
def simple_search_space():
    input = Tensor(name='input', shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    weight = Tensor(name='weight', shape=(32, 3, 1, 1), axis=("O", "I", "kH", "kW"))

    conv = Conv2d(stride=2, dilation=1)   # Define operator and parametrization
    graph = conv(input, weight)           # Define/create/extend graph
    return graph
graph = simple_search_space()
graph
```




    Conv2d(simple_search_space_0.Conv2d_0)



A set of basic operators is implemented in HANNAH, among them 

* Convolution (1D, 2D)
* Linear
* BatchNorm
* Relu
* Add

and more operators will be added in the future. It is also easy to 
define a new operator, see [Custom Ops](#custom-ops). 

## Parametrization & Expressions

!!! note
    For more information about parametrization and expressions, see [Parametrization](parametrization.md).

To build a search space it is not sufficient to feed scalar values to operator parameters. Instead, one can use 
*parameters*. 


```python
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter

@search_space
def simple_parametrized_search_space():
    input = Tensor(name='input', shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    weight = Tensor(name='weight', shape=(IntScalarParameter(min=8, max=64, name='out_channels'), 3, 1, 1), axis=("O", "I", "kH", "kW"))

    # a search space with stride 1 and stride 2 convolutions
    graph = Conv2d(stride=CategoricalParameter(name='stride', choices=[1, 2]))(input, weight)
    return graph
graph = simple_parametrized_search_space()
graph.parametrization(flatten=True)
```




    {'simple_parametrized_search_space_0.Conv2d_0.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = simple_parametrized_search_space_0.Conv2d_0.stride, _registered = True, choices = [1, 2], current_value = 2),
     'simple_parametrized_search_space_0.Conv2d_0.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = simple_parametrized_search_space_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8)}



As futher explained in [Parametrization](parametrization.md), parameters are *expressions* and can be combined to more complex *expressions*,
encoding properties of the search space symbolically. One common use-case is symbolically expressing shapes. Consider for example the following:


```python
in_channel = 3
kernel_size = 1
input = Tensor(name='input',
               shape=(1, in_channel, 32, 32),
               axis=('N', 'C', 'H', 'W'))

@search_space
def simple_search_space(input):
    weight_0 = Tensor(name='weight',
                      shape=(IntScalarParameter(min=8, max=64, name='out_channels'), in_channel, kernel_size, kernel_size),
                      axis=("O", "I", "kH", "kW"))
    conv_0 = Conv2d(stride=CategoricalParameter(name='stride', choices=[1, 2]))(input, weight_0)
    return conv_0
out = simple_search_space(input)
```

How can we know the output shape of `conv_0`, e.g., to put it into the weight tensor of a following convolution, without knowing what value 
the ``out_channel`` parameter has? 
--> Each node has a method `.shape()` which returns the shape as an expression and can be used interchangeably with actual values. Those expressions
are then only evaluated at sampling and during the forward. 


```python
print("Input shape: ", input.shape())
print("Weight shape: ", out.operands[1].shape())
print("Convolution output shape:", out.shape())
```

    Input shape:  (1, 3, 32, 32)
    Weight shape:  (IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = simple_search_space_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8), 3, 1, 1)
    Convolution output shape: (1, IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = simple_search_space_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8), <hannah.nas.expressions.arithmetic.Floor object at 0x7fcaff5b3d60>, <hannah.nas.expressions.arithmetic.Floor object at 0x7fcaff5b2560>)


The `lazy` keyword can be used to evaluate values which *might* be parameters (but could also be `int` or else).


```python
from hannah.nas.functional_operators.lazy import lazy


print("Input shape: ", [lazy(i) for i in input.shape()])
print("Weight shape: ", [lazy(i) for i in out.operands[1].shape()])
print("Convolution output shape:", [lazy(i) for i in out.shape()])
```

    Input shape:  [1, 3, 32, 32]
    Weight shape:  [8, 3, 1, 1]
    Convolution output shape: [1, 8, 16, 16]


When defining an operator, one also has to define a `shape` function (the default shape function is identity, i.e., ``output_shape == input_shape``). Tensors return their own shape. 

## Graphs and Hierarchy

As seen in the simple examples above, we can chain op and tensor nodes together to create graphs and use parameters to span search spaces.


```python
from hannah.nas.functional_operators.operators import Relu


@search_space
def simple_search_space():
    input = Tensor(name='input',
                   shape=(1, 3, 32, 32),
                   axis=('N', 'C', 'H', 'W'))

    weight_0 = Tensor(name='weight', shape=(IntScalarParameter(min=8, max=64, name='out_channels'), 3, 1, 1), axis=("O", "I", "kH", "kW"))

    conv_0 = Conv2d(stride=CategoricalParameter(name='stride', choices=[1, 2]))(input, weight_0)
    relu_0 = Relu()(conv_0)

    weight_1 = Tensor(name='weight', shape=(IntScalarParameter(min=32, max=64, name='out_channels'), conv_0.shape()[1], 3, 3), axis=("O", "I", "kH", "kW"))
    conv_1 = Conv2d(stride=CategoricalParameter(name='stride', choices=[1, 2]))(relu_0, weight_1)
    relu_1 = Relu()(conv_1)
    return relu_1
out = simple_search_space()

```


```python
out.parametrization(flatten=True)
```




    {'simple_search_space_0.Conv2d_1.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = simple_search_space_0.Conv2d_1.stride, _registered = True, choices = [1, 2], current_value = 2),
     'simple_search_space_0.Conv2d_1.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = simple_search_space_0.Conv2d_1.weight.out_channels, _registered = True, min = 32, max = 64, step_size = 1, current_value = 32),
     'simple_search_space_0.Conv2d_0.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = simple_search_space_0.Conv2d_0.stride, _registered = True, choices = [1, 2], current_value = 2),
     'simple_search_space_0.Conv2d_0.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = simple_search_space_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8)}



Nodes have *operands* for backwards traversal and *users* for forward traversal.
With helper functions like `get_nodes` one can iterate through all graph nodes.


```python
from hannah.nas.functional_operators.op import get_nodes

print("Relu Operands: ", out.operands)
print("Conv Users: ", out.operands[0].users)

print("\nNodes:")
for node in get_nodes(out):
    print('Node:', node)
    print('\tOperands: ', node.operands)

```

    Relu Operands:  [Conv2d(simple_search_space_0.Conv2d_1)]
    Conv Users:  [Relu(simple_search_space_0.Relu_1)]
    
    Nodes:
    Node: Relu(simple_search_space_0.Relu_1)
    	Operands:  [Conv2d(simple_search_space_0.Conv2d_1)]
    Node: Conv2d(simple_search_space_0.Conv2d_1)
    	Operands:  [Relu(simple_search_space_0.Relu_0), Tensor(simple_search_space_0.Conv2d_1.weight)]
    Node: Tensor(simple_search_space_0.Conv2d_1.weight)
    	Operands:  []
    Node: Relu(simple_search_space_0.Relu_0)
    	Operands:  [Conv2d(simple_search_space_0.Conv2d_0)]
    Node: Conv2d(simple_search_space_0.Conv2d_0)
    	Operands:  [Tensor(simple_search_space_0.input), Tensor(simple_search_space_0.Conv2d_0.weight)]
    Node: Tensor(simple_search_space_0.Conv2d_0.weight)
    	Operands:  []
    Node: Tensor(simple_search_space_0.input)
    	Operands:  []


### Blocks

Creating large graphs with a lot of operators and tensors manually can get tedious and convoluted. Instead, we can define search space graphs in a hierarchical manner by encapsulating them in functions:


```python
def conv_relu(input, kernel_size, out_channels, stride):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'),
                    grad=True)

    conv = Conv2d(stride=stride)(input, weight)
    relu = Relu()(conv)
    return relu
```


```python
input = Tensor(name='input',
               shape=(1, 3, 32, 32),
               axis=('N', 'C', 'H', 'W'))
@search_space
def space(input):
    kernel_size = CategoricalParameter(name="kernel_size", choices=[1, 3, 5])
    stride = CategoricalParameter(name="stride", choices=[1, 2])
    out_channels = IntScalarParameter(name="out_channels", min=8, max=64)
    net = conv_relu(input, kernel_size=kernel_size, out_channels=out_channels, stride=stride)
    net = conv_relu(net, kernel_size=kernel_size, out_channels=out_channels, stride=stride)
    return net

net = space(input)

for n in get_nodes(net):
    print(n)
```

    Relu(space_0.Relu_1)
    Conv2d(space_0.Conv2d_1)
    Tensor(space_0.Conv2d_1.weight)
    Relu(space_0.Relu_0)
    Conv2d(space_0.Conv2d_0)
    Tensor(space_0.Conv2d_0.weight)
    Tensor(input)



```python
net.parametrization(flatten=True)
```




    {'space_0.Conv2d_0.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = space_0.Conv2d_0.stride, _registered = True, choices = [1, 2], current_value = 1),
     'space_0.Conv2d_0.weight.kernel_size': CategoricalParameter(rng = Generator(PCG64), name = kernel_size, id = space_0.Conv2d_0.weight.kernel_size, _registered = True, choices = [1, 3, 5], current_value = 1),
     'space_0.Conv2d_0.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = space_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8)}



Note, how there is just one set of parameters. If defined this way, both blocks share their parameters. To define seperate parameters one can use `param.new()`


```python
input = Tensor(name='input',
               shape=(1, 3, 32, 32),
               axis=('N', 'C', 'H', 'W'))
@search_space
def space(input):
    kernel_size = CategoricalParameter(name="kernel_size", choices=[1, 3, 5])
    stride = CategoricalParameter(name="stride", choices=[1, 2])
    out_channels = IntScalarParameter(name="out_channels", min=8, max=64)
    net = conv_relu(input, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    net = conv_relu(net, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    return net
net = space(input)

net.parametrization(flatten=True)
```




    {'space_0.Conv2d_1.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = space_0.Conv2d_1.stride, _registered = True, choices = [1, 2], current_value = 2),
     'space_0.Conv2d_1.weight.kernel_size': CategoricalParameter(rng = Generator(PCG64), name = kernel_size, id = space_0.Conv2d_1.weight.kernel_size, _registered = True, choices = [1, 3, 5], current_value = 5),
     'space_0.Conv2d_1.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = space_0.Conv2d_1.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8),
     'space_0.Conv2d_0.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = space_0.Conv2d_0.stride, _registered = True, choices = [1, 2], current_value = 2),
     'space_0.Conv2d_0.weight.kernel_size': CategoricalParameter(rng = Generator(PCG64), name = kernel_size, id = space_0.Conv2d_0.weight.kernel_size, _registered = True, choices = [1, 3, 5], current_value = 5),
     'space_0.Conv2d_0.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = space_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8)}



These function blocks can be nested as desired.


```python
def block(input):
    kernel_size = CategoricalParameter(name="kernel_size", choices=[1, 3, 5])
    stride = CategoricalParameter(name="stride", choices=[1, 2])
    out_channels = IntScalarParameter(name="out_channels", min=8, max=64)
    net = conv_relu(input, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    net = conv_relu(net, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    net = conv_relu(net, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    return net

input = Tensor(name='input',
               shape=(1, 3, 32, 32),
               axis=('N', 'C', 'H', 'W'))
@search_space
def space(input):
    net = block(input)
    net = block(net)
    return net
net = space(input)

for n in get_nodes(net):
    print(n)
```

    Relu(space_0.Relu_5)
    Conv2d(space_0.Conv2d_5)
    Tensor(space_0.Conv2d_5.weight)
    Relu(space_0.Relu_4)
    Conv2d(space_0.Conv2d_4)
    Tensor(space_0.Conv2d_4.weight)
    Relu(space_0.Relu_3)
    Conv2d(space_0.Conv2d_3)
    Tensor(space_0.Conv2d_3.weight)
    Relu(space_0.Relu_2)
    Conv2d(space_0.Conv2d_2)
    Tensor(space_0.Conv2d_2.weight)
    Relu(space_0.Relu_1)
    Conv2d(space_0.Conv2d_1)
    Tensor(space_0.Conv2d_1.weight)
    Relu(space_0.Relu_0)
    Conv2d(space_0.Conv2d_0)
    Tensor(space_0.Conv2d_0.weight)
    Tensor(input)


### Scopes

As seen above, while the *definition* of the graph is made in a hierarchical manner, the actual graph and its node are "flat" and do not have any inherent hierarchy. To make the graph more clear and readable one can use **scopes** with the `@scope` decorator for blocks. Note that `@scope` does not have any effect on the inherent structure of the graph but only affects the node `id`s.


```python
from hannah.nas.functional_operators.op import scope


@scope
def conv_relu(input, kernel_size, out_channels, stride):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'),
                    grad=True)

    conv = Conv2d(stride=stride)(input, weight)
    relu = Relu()(conv)
    return relu

@scope
def block(input):
    kernel_size = CategoricalParameter(name="kernel_size", choices=[1, 3, 5])
    stride = CategoricalParameter(name="stride", choices=[1, 2])
    out_channels = IntScalarParameter(name="out_channels", min=8, max=64)
    net = conv_relu(input, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    net = conv_relu(net, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    net = conv_relu(net, kernel_size=kernel_size.new(), out_channels=out_channels.new(), stride=stride.new())
    return net


input = Tensor(name='input',
               shape=(1, 3, 32, 32),
               axis=('N', 'C', 'H', 'W'))
@search_space
def space(input):
    net = block(input)
    net = block(net)
    return net
net = space(input)

for n in get_nodes(net):
    print(n)
```

    Relu(space_0.block_1.conv_relu_2.Relu_0)
    Conv2d(space_0.block_1.conv_relu_2.Conv2d_0)
    Tensor(space_0.block_1.conv_relu_2.Conv2d_0.weight)
    Relu(space_0.block_1.conv_relu_1.Relu_0)
    Conv2d(space_0.block_1.conv_relu_1.Conv2d_0)
    Tensor(space_0.block_1.conv_relu_1.Conv2d_0.weight)
    Relu(space_0.block_1.conv_relu_0.Relu_0)
    Conv2d(space_0.block_1.conv_relu_0.Conv2d_0)
    Tensor(space_0.block_1.conv_relu_0.Conv2d_0.weight)
    Relu(space_0.block_0.conv_relu_2.Relu_0)
    Conv2d(space_0.block_0.conv_relu_2.Conv2d_0)
    Tensor(space_0.block_0.conv_relu_2.Conv2d_0.weight)
    Relu(space_0.block_0.conv_relu_1.Relu_0)
    Conv2d(space_0.block_0.conv_relu_1.Conv2d_0)
    Tensor(space_0.block_0.conv_relu_1.Conv2d_0.weight)
    Relu(space_0.block_0.conv_relu_0.Relu_0)
    Conv2d(space_0.block_0.conv_relu_0.Conv2d_0)
    Tensor(space_0.block_0.conv_relu_0.Conv2d_0.weight)
    Tensor(input)


## Choice Ops

A choice op is a special node kind that allows to have multiple paths in the graph that exclude each other (or have other specialized behaviour). 


```python
from hannah.nas.functional_operators.operators import Identity
from functools import partial
from hannah.nas.functional_operators.op import ChoiceOp

@scope
def choice_block(input):
    kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
    out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
    stride = CategoricalParameter([1, 2], name='stride')

    identity = Identity()
    optional_conv = partial(conv_relu, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
    net = ChoiceOp(identity, optional_conv)(input)
    return net


```


```python
kernel_size = CategoricalParameter(name="kernel_size", choices=[1, 3, 5])
stride = CategoricalParameter(name="stride", choices=[1, 2])
out_channels = IntScalarParameter(name="out_channels", min=8, max=64)


input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))

@search_space
def space(input, out_channels, stride, kernel_size):
    conv = conv_relu(input, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
    net = choice_block(conv)
    return net
net = space(input, out_channels, stride, kernel_size)

net.parametrization(flatten=True)

```




    {'space_0.choice_block_0.ChoiceOp_0.choice': IntScalarParameter(rng = Generator(PCG64), name = choice, id = space_0.choice_block_0.ChoiceOp_0.choice, _registered = True, min = 0, max = 1, step_size = 1, current_value = 0),
     'space_0.choice_block_0.conv_relu_0.Conv2d_0.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = space_0.choice_block_0.conv_relu_0.Conv2d_0.stride, _registered = True, choices = [1, 2], current_value = 1),
     'space_0.choice_block_0.conv_relu_0.Conv2d_0.weight.kernel_size': CategoricalParameter(rng = Generator(PCG64), name = kernel_size, id = space_0.choice_block_0.conv_relu_0.Conv2d_0.weight.kernel_size, _registered = True, choices = [1, 3, 5], current_value = 1),
     'space_0.choice_block_0.conv_relu_0.Conv2d_0.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = space_0.choice_block_0.conv_relu_0.Conv2d_0.weight.out_channels, _registered = True, min = 4, max = 64, step_size = 1, current_value = 4),
     'space_0.conv_relu_0.Conv2d_0.stride': CategoricalParameter(rng = Generator(PCG64), name = stride, id = space_0.conv_relu_0.Conv2d_0.stride, _registered = True, choices = [1, 2], current_value = 1),
     'space_0.conv_relu_0.Conv2d_0.weight.kernel_size': CategoricalParameter(rng = Generator(PCG64), name = kernel_size, id = space_0.conv_relu_0.Conv2d_0.weight.kernel_size, _registered = True, choices = [1, 3, 5], current_value = 3),
     'space_0.conv_relu_0.Conv2d_0.weight.out_channels': IntScalarParameter(rng = Generator(PCG64), name = out_channels, id = space_0.conv_relu_0.Conv2d_0.weight.out_channels, _registered = True, min = 8, max = 64, step_size = 1, current_value = 8)}



![Choice node](../../assets/choice_node.jpg)

!!! note
    When defining options for a choice node, one can either use ops directly (see ``Identity()`` above) or use block functions (``conv_relu``). For block functions, one has to use ``functools.partial`` to enable 
    the choice node to perform the respective integration in the graph.  

During execution, the choice node can be leveraged to define the behaviour (e.g., select one and only one path, execute all paths and return a parametrized sum for differential NAS, ...). Choice nodes can, for example, be used to search over different operator types, different operator patterns, or to implement dynamic depth/a variable amount of layers/blocks.  


```python
def dynamic_depth(*exits, switch):
    return ChoiceOp(*exits, switch=switch)()
```

## Custom Ops

To define custom operators, one can inherit from the ``Op`` class. Then, one can override the ``__call__(self, *operands)`` class to perform specific actions, e.g., saving certain parameters of the operands as fields of the operator instance that is returned. Don't forget to call ``super().__call__(*operands)``, which performs the integration of the new operator instance into the graph. 

Then, one has to provide a ``_forward_implementation(self, *args)``, which defines the computation that the operator executes. 

Lastly, a ``shape_fun(self)`` defines the output shape of the operator.

## Executor

The search space graphs are not themselves executable. For that one needs an ``Executor``. The ``BasicExecutor`` analyzes the graph to find dependencies and a valid node order (e.g., to execute the results of operands first before they are added in an ``Add`` operation) and builds a ``forward`` function. It also registers torch parameters and buffers for training.The executor should be usable as a normal ``torch.nn.Module``. One can define custom executors, e.g., for weight sharing NAS or differential NAS. 


```python
import torch
from hannah.nas.functional_operators.executor import BasicExecutor


input = Tensor(name='input',
               shape=(1, 3, 32, 32),
               axis=('N', 'C', 'H', 'W'))
@search_space
def space(input):
    net = block(input)
    net = block(net)
    return net
net = space(input)
model = BasicExecutor(net)
model.initialize()

x = torch.randn(input.shape())
model.forward(x)
```




    tensor([[[[0.2717, 0.0092, 0.1203, 0.1979],
              [0.0000, 0.2005, 0.0972, 0.0256],
              [0.1351, 0.1363, 0.0754, 0.1609],
              [0.0000, 0.1031, 0.0446, 0.2227]],
    
             [[0.2462, 0.0013, 0.0224, 0.0534],
              [0.2030, 0.1310, 0.0000, 0.0404],
              [0.1303, 0.1276, 0.0634, 0.1498],
              [0.1786, 0.0298, 0.0085, 0.1301]],
    
             [[0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000]],
    
             [[0.0000, 0.0021, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0232, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0011, 0.0000]],
    
             [[0.7481, 0.0018, 0.2029, 0.1693],
              [0.7117, 0.3248, 0.1578, 0.1085],
              [0.3086, 0.3926, 0.1606, 0.3065],
              [0.5410, 0.1157, 0.0583, 0.4534]],
    
             [[0.0000, 0.0000, 0.0705, 0.0628],
              [0.0000, 0.0000, 0.1682, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0381, 0.0255, 0.0000]],
    
             [[0.7549, 0.0092, 0.2340, 0.1351],
              [0.7965, 0.1582, 0.2039, 0.0925],
              [0.2619, 0.3976, 0.1461, 0.1876],
              [0.5799, 0.0848, 0.0732, 0.4952]],
    
             [[0.5984, 0.0043, 0.2075, 0.1700],
              [0.5905, 0.1869, 0.2142, 0.0772],
              [0.2146, 0.3152, 0.1176, 0.1768],
              [0.4285, 0.1043, 0.0665, 0.3872]]]], grad_fn=<ReluBackward0>)




```python

```
