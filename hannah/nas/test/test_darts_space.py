from hannah.nas.ops import batched_image_tensor
from hannah.nas.spaces.darts.darts_space import darts_space
from hannah.nas.dataflow.ops import conv2d, identity, sum, concat


def test_darts_space():
    input = batched_image_tensor(name='input')
    darts  = darts_space(input)
    print()


if __name__ == '__main__':
    test_darts_space()