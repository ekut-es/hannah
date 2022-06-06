import unittest
import torch

import torch.nn as nn
import torch.nn.functional as F


class MyTestCase(unittest.TestCase):
    # def test_something(self):

    #     input = torch.tensor([
    #         [[1.],
    #          [2.],
    #          [3.],
    #          [4.]]
    #     ])
    #     print(input.shape)
    #     print(input)
    #     weights = torch.tensor([
    #         [
    #             [10.],
    #             [10.]
    #         ],
    #         [
    #             [20.],
    #             [20.]
    #         ],
    #         [
    #             [30.],
    #             [30.]
    #         ],
    #         [
    #             [40.],
    #             [40.]
    #         ],
    #     ])
    #     print(weights.shape)
    #     print(weights.shape[0]*weights.shape[1] * weights.shape[2])

    #     res = F.conv1d(input, weights, groups=2)
    #     print(res)
    #     print(res.shape)

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
            m.weight = nn.Parameter(self.adjust_weights_for_grouping(m.weight, m.groups))
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
        m_diff = m.groups - m_groups_old

        with torch.no_grad():
            m.weight = nn.Parameter(self.adjust_weights_for_grouping(m.weight, m_diff))
            weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]

        assert list(m.weight.shape) == weight_shape
        output3 = m(input)

        print(output.shape)
        print(output2.shape)
        print(output3.shape)

        assert output.shape == output2.shape and output.shape == output3.shape
        print("test ending")

    def adjust_weights_for_grouping(self, weights, groups):
        """
        Adjusts the Weights for the Forward of the Convulution
        Shape(outchannels, inchannels / group, kW)
        weight – filters of shape (\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)(out_channels,
        groups
        in_channels
         ,kW)
        """

        # logging.info(f"Weights shape is {weights.shape}")
        # torch.reshape(weights, [weights.shape[0], weights.shape[1] / group, weights.shape[2]])
        # input_shape : int = np.floor(weights.shape[1] / group).astype(int)
        # hier rausschneiden oder maskieren

        channels_per_group = weights.shape[1] // groups

        splitted_weights = torch.tensor_split(weights, groups)
        result_weights = []

        # for current_group in range(groups):
        for current_group, current_weight in enumerate(splitted_weights):
            input_start = current_group * channels_per_group
            input_end = input_start + channels_per_group
            current_result_weight = current_weight[:, input_start:input_end, :]
            result_weights.append(current_result_weight)

        full_kernel = torch.concat(result_weights)

        print(full_kernel.shape)
        return full_kernel


if __name__ == "__main__":
    unittest.main()
