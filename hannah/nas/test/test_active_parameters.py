from hannah.nas.functional_operators.op import Tensor
from hannah.models.embedded_vision_net.models import embedded_vision_net
from hannah.nas.functional_operators.utils import get_active_parameters


def test_active_parameters():
    input = Tensor(name="input", shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    space = embedded_vision_net("space", input, num_classes=10)
    space.parametrization()["embedded_vision_net_0.ChoiceOp_0.num_blocks"].set_current(1)
    space.parametrization()["embedded_vision_net_0.block_0.pattern_0.ChoiceOp_0.choice"].set_current(4)
    space.parametrization()["embedded_vision_net_0.block_0.pattern_0.sandglass_block_0.expansion_0.ChoiceOp_0.choice"].set_current(1)
    active_params = get_active_parameters(space)

    space.parametrization()
    print()


if __name__ == "__main__":
    test_active_parameters()