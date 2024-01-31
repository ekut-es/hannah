from hannah.models.embedded_vision_net.blocks import grouped_pointwise
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import Tensor
from torch.testing import assert_close
import torch.nn as nn
import torch


def test_grouped_pointwise():
    class GroupedPW(nn.Module):
        def __init__(self, in_channels, step_size=2, groups=2, out_channels=128) -> None:
            super().__init__()
            self.groups = groups
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.step_size = step_size

            self.pw_k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=self.groups)
            self.pw_l = nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=self.groups)

        def forward(self, x):
            out_0 = self.pw_k(x)
            out = torch.concat([out_0[:, shift_pos::self.step_size, : , :] for shift_pos in range(self.step_size)], dim=1)
            out = self.pw_l(out)
            out = torch.add(out_0, out)
            return out

    input = Tensor(name="input", shape=(1, 64, 32, 32), axis=("N", "C", "H", "W"))
    grouped_pw = grouped_pointwise(input, out_channels=128)
    model = BasicExecutor(grouped_pw)
    model.initialize()

    x = torch.ones(input.shape())
    out = model(x)
    groups = grouped_pw.operands[0].groups.evaluate()
    torch_mod = GroupedPW(in_channels=64, step_size=groups, groups=groups, out_channels=128)

    params = dict(model.named_parameters())
    with torch.no_grad():
        torch_mod.pw_k.weight = params["grouped_pointwise_0_Conv2d_0_weight"]
        torch_mod.pw_l.weight = params["grouped_pointwise_0_Conv2d_1_weight"]

    torch_out = torch_mod(x)

    assert_close(out, torch_out)


if __name__ == '__main__':
    test_grouped_pointwise()
