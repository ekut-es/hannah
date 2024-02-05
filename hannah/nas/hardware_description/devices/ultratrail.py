from hannah.nas.expressions.placeholder import IntRange, UndefinedFloat, UndefinedInt
from hannah.nas.functional_operators.op import Tensor, scope
from hannah.nas.functional_operators.operators import Add, Conv2d, Quantize, Relu


@scope
def ut_op(
    weight_bits: int = 8,
    bias_bits: int = 8,
    activation_bits: int = 8,
    accumulator_bits: int = 8,
    max_weight_bits: int = 8,
    max_kernel_size: int = 2**4,
    max_input_length: int = 2**7,
    max_input_channel_block: int = 2**4,
    max_output_channel_block: int = 2**4,
    stride_range: int = 2**3,
):
    input_data_type = IntType(signed=True, bits=activation_bits)
    input_quantization = quantization(scale=UndefinedFloat(), zero_point=0)

    input = Tensor(name='input',
                   shape=(UndefinedInt(), UndefinedInt(), UndefinedInt()),
                   axis=('N', 'C', 'W'),
                   dtype=input_data_type)

    weight_data_type = IntType(signed=True, bits=weight_bits)
    weight_quantization = quantization(scale=UndefinedFloat(), zero_point=0)

    weight = Tensor(name="weight",
                    axis=('O', 'I', 'kW'),
                    shape=(UndefinedInt(), UndefinedInt(), IntRange(1, max_kernel_size)),
                    dtype=weight_data_type)

    res_input = Tensor(name='res_input',
                       shape=(UndefinedInt(), UndefinedInt(), UndefinedInt()),
                       axis=('N', 'C', 'W'),
                       dtype=input_data_type)

    conv_out = conv1d(input, weight, IntRange(1, stride_range))

    accumulator_data_type = IntType(signed=True, bits=accumulator_bits)
    accumulator_quantization = quantization(scale=UndefinedFloat(), zero_point=0)

    requant_conv = quantize(conv_out, accumulator_data_type, accumulator_quantization)

    bias_data_type = IntType(signed=True, bits=bias_bits)
    bias = Tensor(name="bias",
                  shape=(UndefinedInt()),
                  axis=("C"),
                  dtype=bias_data_type)

    bias_add = add(requant_conv, bias)
    out = optional(bias_add, requant_conv)

    res_add = add(out, res_input)
    out = optional(res_add, out)

    pool = avg_pool(out)
    out = optional(pool, out)

    activation = relu(pool)
    out = optional(activation, out)

    requantization = quantize(out, input_data_type, input_quantization)

    return requantization






