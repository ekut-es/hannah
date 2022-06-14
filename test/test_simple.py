import unittest
import torch

import torch.nn as nn
import torch.nn.functional as F

from hannah.models.ofa.utilities import (adjust_weights_for_grouping)


class MyTestCase(unittest.TestCase):
    def test_something(self):

        input = torch.tensor([
            [[1.],
             [2.],
             [3.],
             [4.]]
        ])
        print(input.shape)
        print(input)
        weights = torch.tensor([
            [
                [10.],
                [10.]
            ],
            [
                [20.],
                [20.]
            ],
            [
                [30.],
                [30.]
            ],
            [
                [40.],
                [40.]
            ],
        ])
        print(weights.shape)
        print(weights.shape[0]*weights.shape[1] * weights.shape[2])

        res = F.conv1d(input, weights, groups=2)
        print(res)
        print(res.shape)

    def test_compareGroups(self):
        m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=1)
        n = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=2)
        input = torch.randn(10, 16, 100)  # batch, input, c_out
        output = m(input)
        output2 = n(input)
        print(f"Weights {m.weight.shape}")
        print(f"Weights {n.weight.shape}")
        print(output.shape)
        print(output2.shape)
        # print(output)
        print("test ending")

    def test_weights(self):
        m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=1)
        m_size = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
        input = torch.randn(10, 16, 100)  # batch, input, c_out
        output = m(input)
        m.groups = 4
        with torch.no_grad():
            m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m.groups))
        print(f"Weights {m.weight.shape}")
        m_size_2 = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
        output2 = m(input)

        print(f"Weight_Sizes before:{m_size} after:{m_size_2}")
        print(output.shape)
        print(output2.shape)
        # print(output)
        print("test ending")

    # def test_adjust_and_restore(self):
    #     m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=1)
    #     weight_shape = list(m.weight.shape)
    #     input = torch.randn(10, 16, 100)  # batch, input, c_out
    #     output = m(input)
    #     m.groups = 4
    #     with torch.no_grad():
    #         m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m.groups))
    #     print(f"Weights {m.weight.shape}")
    #     output2 = m(input)
    #     m.groups = 1
    #     with torch.no_grad():
    #         new_weight = self.restore_shape_weights(m.weight, 4)
    #         m.weight = nn.Parameter(new_weight)
    #     print(f"Weights {m.weight.shape}")
    #     assert weight_shape == list(m.weight.shape)
    #     print(output.shape)
    #     print(output2.shape)
    #     # print(output)
    #     print("test ending")

    def test_sequencedGrouping(self):
        """
            Name will be upgraded later
            It is a solution to the problem, that the grouping cannot be applied
            in random steps (so far).

            If we first have groups=6 and then groups=8 we can't just reshape the 'grouped=6' weights to 'groups=8' shape (based on grouped=6 weights)

            Though it is possible, that we have a sequenced grouping phase like 2,4,8,16,32 (which would be also conveniant for quantizied versions?)
            This approach will be demonstrated in that unit case
        """
        kernel_size = 3

        out_channels = 32
        in_channels = 16
        groups = [i for i in range(1, out_channels) if in_channels % i == 0]
        groups.sort(reverse=False)

        print(groups)

        m = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups[0]
        )  # changes
        weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
        input = torch.randn(10, 16, 100)  # batch, input, c_out

        output_before = m(input)
        print(f"Weights {m.weight.shape}")
        print(output_before.shape)
        assert list(m.weight.shape) == weight_shape
        for i in groups:
            if(i == 1):
                continue
            m.groups = i
            with torch.no_grad():
                m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, 2))
                weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
                assert list(m.weight.shape) == weight_shape
            output = m(input)
            assert list(m.weight.shape) == weight_shape

        assert output_before.shape == output.shape
        print("test ending")

    def test_changeGroups(self):
        kernel_size = 3
        m = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, groups=1
        )  # changes
        weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
        input = torch.randn(10, 16, 100)  # batch, input, c_out
        output = m(input)
        print(f"Weights {m.weight.shape}")
        print(output.shape)
        assert list(m.weight.shape) == weight_shape
        m.groups = 2
        with torch.no_grad():
            m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m.groups))
            weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
        assert list(m.weight.shape) == weight_shape

        output2 = m(input)

        # Die Mehrmalige Teilung funktioniert hier nicht.
        # Hier teilt er die Gewichte erneut durch 4 statt von ursprünglichen Gewichtsverhältnis
        # Hierbei sind zwei Wege möglich :
        # Gewichte durch die Differenz teilen
        # Ursprüngliche Gewichtsdimension merken und anhand derer die Gewichte teilen
        # Version eins
        m_groups_old = m.groups
        m.groups = 4
        # Does not work with 8 - so this is not the solution
        m_diff = m.groups - m_groups_old

        with torch.no_grad():
            m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m_diff))
            weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]

        assert list(m.weight.shape) == weight_shape
        output3 = m(input)

        print(output.shape)
        print(output2.shape)
        print(output3.shape)

        assert output.shape == output2.shape and output.shape == output3.shape
        print("test ending")


if __name__ == "__main__":
    unittest.main()
