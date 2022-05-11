from typing import Optional, Tuple

from torch import Tensor
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.compression_type import CompressionType
from hannah.nas.dataflow.data_type import DataType, FloatType
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.quantization_type import QuantizationType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.hardware_description.memory_type import MemoryType
from hannah.nas.parameters import IntScalarParameter
from hannah.nas.parameters.parameters import CategoricalParameter


def test_dataflow():
    def axis(name: str, size: Optional[int] = None, compression: Optional[CompressionType] = None):
        return AxisType(name=name, size=size, compression=compression)


    def tensor(axis: Tuple[AxisType, ...],
               dtype: DataType = FloatType(),
               quantization: Optional[QuantizationType] = None,
               memory: Optional[MemoryType] = None):
        return TensorType(axis=axis, dtype=dtype, quantization=quantization, memory=memory)

    @dataflow
    def conv(input: TensorType,
             output_channel=IntScalarParameter(4, 64),
             kernel_size=CategoricalParameter([1, 3, 5]),
             stride=CategoricalParameter([1, 2])):

        weight = tensor((axis('o', size=output_channel),
                        axis('i', size=input.axis['c'].size),
                        axis('kh', size=kernel_size),
                        axis('kw', size=kernel_size)))
        return OpType('conv2d', input, weight, stride=stride)

    @dataflow
    def conv_block():
        conv
        conv
        conv


    input = tensor((axis('n'),
                    axis('c'),
                    axis('h'),
                    axis('w')))

    out = conv(input)
    print(out)



if __name__ == '__main__':
    test_dataflow()