from hannah.nas.ops import batched_image_tensor
from hannah.nas.spaces.mobilenet.mobilenet import mobilenet



def test_mbn_space():
    input = batched_image_tensor(name='input')
    space  = mobilenet(input, num_cells=4)
    print()


if __name__ == '__main__':
    test_mbn_space()
