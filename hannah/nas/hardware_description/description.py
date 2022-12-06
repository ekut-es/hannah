from typing import Optional, Tuple

from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.compression_type import CompressionType
from hannah.nas.dataflow.data_type import DataType, FloatType, IntType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.optional_op import OptionalOp
from hannah.nas.dataflow.quantization_type import QuantizationType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.hardware_description.device import Ultratrail
from hannah.nas.hardware_description.memory_type import MemoryType
from hannah.nas.parameters import IntScalarParameter

if __name__ == "__main__":
    ultratrail = Ultratrail(
        weight_bits=IntScalarParameter(min=1, max=8),
        bias_bits=IntScalarParameter(min=1, max=8),
        activation_bits=IntScalarParameter(min=1, max=8),
        accumulator_bits=IntScalarParameter(min=1, max=32),
        max_weight_bits=IntScalarParameter(min=4, max=8),
    )

    print(ultratrail)
