from ...core.expression import Expression
from ...ops import relu, swish, leaky_relu

def act(input: TensorType, activation_type: Expression[str]):
    act = select(activation_type, relu=relu(input), swish=swish(input). leaky_relu=leaky_relu(input))

    return Output(act)