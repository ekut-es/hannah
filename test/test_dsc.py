from tokenize import group
import unittest
import torch
import pytest

import torch.nn as nn
import torch.nn.functional as F

from hannah.models.ofa.utilities import (
    create_channel_filter,
    prepare_kernel_for_depthwise_separable_convolution,
    prepare_kernel_for_pointwise_convolution,
)


class Test_DSC(unittest.TestCase):

    def test_dsc(self):

        input = torch.randn(10, 16, 100)  # batch, input, c_out

        in_channel = 16
        out_channel = 32
        kernel_size = 3
        groups = 4
        compare = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, groups=groups)

        print(input.shape)
        t = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=1,
        )
        full_kernel = t.weight.data
        depthwise_output_filter = create_channel_filter(
            t,
            full_kernel,
            current_channel=out_channel,
            reduced_target_channel_size=in_channel,
        )
        input_filter = create_channel_filter(
            t,
            full_kernel,
            current_channel=in_channel,
            reduced_target_channel_size=in_channel,
        )
        new_kernel, bias = prepare_kernel_for_depthwise_separable_convolution(
            kernel=full_kernel,
            bias=None,
            in_channel_count=in_channel,
            in_channel_filter=input_filter,
            out_channel_filter=depthwise_output_filter,
        )
        print(
            f"kernel:{new_kernel.shape} bias: {bias.shape if bias is not None else bias}"
        )
        # perform depthwise separable convolution
        res_depthwise = F.conv1d(input, new_kernel, bias, groups=in_channel)
        # point_conv = Conv2d(in_channels=10, out_channels=32, kernel_size=1)

        assert res_depthwise.shape[1] == in_channel
        print(res_depthwise)
        print(res_depthwise.shape)
        # get new kernel size
        # use full kernel
        point_wise_output_filter = create_channel_filter(
            t,
            full_kernel,
            current_channel=out_channel,
            reduced_target_channel_size=out_channel,
        )
        # grouping = in_channel_count
        new_kernel, bias = prepare_kernel_for_pointwise_convolution(
            kernel=full_kernel,
            grouping=groups,
        )
        res_pointwise = F.conv1d(res_depthwise, new_kernel, bias, groups=groups)
        assert res_pointwise.shape[1] == out_channel
        assert new_kernel.shape[2] == 1
        print(res_pointwise)
        print(res_pointwise.shape)

        print("Comparing with normal conv")
        compare_output = compare(input)
        print(f"dsc:{res_pointwise.shape}, compare:{compare_output.shape}")
        assert(compare_output.shape == res_pointwise.shape)
        # TODO Next Step call from Elastic Kernel Conv

    @pytest.mark.skip(reason="only for looking stuff up")
    def test_kernel_dimensions(self):
        t = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3)
        a = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1)
        print(t.weight.shape)
        print(t.weight.size(0))
        print(t.weight.size(1))
        print(t.weight.size(2))
        print("=================")
        print(a.weight.shape)
        print(a.weight.size(0))
        print(a.weight.size(1))
        print(a.weight.size(2))
        print("=================")
        print(t.weight.shape[0] * t.weight.shape[1] * t.weight.shape[2])
        print(a.weight.shape[0] * a.weight.shape[1] * a.weight.shape[2])
        print("=================")
        print(t.weight)
        print(a.weight)
        print(a.weight[:, :, 0])
        print(t.weight[:, :, 0])
        # Bedeutet wenn ich kernel[a][b] mache, dann nehme ich den ersten wert von dem paket
        # bedeutet nimm alle elemente und von der letzten dimension slice immer nur das erste
        # kernel[:,:,0].shape sliced alles so, dass es von allen dims, bei der letzten nur das nullte nimmt

    @pytest.mark.skip(reason="only for looking stuff up")
    # TODO: nächster Schritt, vergleiche Depthwise Separable mit normalen, und zwar input und output.
    def test_kernel_reducement(self):
        input = torch.randn(10, 2, 100)  # batch, input, c_out
        t = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3)
        a = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1)
        t.weight.data = torch.ones(2, 2, 3)
        a.weight.data = torch.ones(2, 2, 1)
        kernel = t.weight.data
        output_1 = F.conv1d(input, kernel)
        kernel_reduced = kernel[:, :, 0:1]
        print(f"kernel_size reduced: {kernel_reduced.shape} before: {kernel} target: {a.weight.shape}")
        assert(kernel_reduced.shape == a.weight.shape)
        output_2 = F.conv1d(input, kernel_reduced)
        output_compare = F.conv1d(input, a.weight.data)
        assert(output_2.shape == output_compare.shape)
        assert(torch.equal(output_2, output_compare))
        print("finished testing")

    @pytest.mark.skip(reason="only for looking stuff up")
    def test_check_weigths(self):
        t = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=1)
        print(f"weight: {t.weight.data.shape}")
        # weight is defined out_channels, in_channels, kernel_size
        print([t.out_channels, t.in_channels, t.kernel_size[0]])

    @pytest.mark.skip(reason="only for looking stuff up")
    def test_combinedGrouping(self):
        t = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=1)
        a = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, groups=1)

        input = torch.randn(10, 16, 100)  # batch, input, c_out

        print("Cnn Combination without grouping")
        printValues(1, t)
        printValues(2, a)
        print(f"with Input: {input.shape}")
        output_without_group = t(input)
        print(f"Output 1 without grouping: {output_without_group.shape}")
        output_without_group2 = a(output_without_group)
        print(f"Output 2 without grouping: {output_without_group2.shape}")
        print("===========")
        print("Weights: \n")
        print(f"Weights of Conv 1:  {t.weight.shape}")
        print(f"Weights of Conv 2:{a.weight.shape}")
        # print(f"C{number} Parameters: in_channels={t.in_channels}, out_channels={t.out_channels}, k={t.kernel_size}, g={t.groups}")
        # print(f"C2 Parameters: in_channels={a.in_channels}, out_channels={a.out_channels}, k={a.kernel_size}, g={a.groups}")
        print("=================================")
        print("Cnn Combination with grouping")
        m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=8)
        n = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, groups=2)
        printValues(1, m)
        printValues(2, n)
        output = m(input)
        print(f"Output1 with grouping: {output.shape}")
        output2 = n(output)
        print(f"Output2 with grouping: {output2.shape}")
        print("===========")
        print("Weights: \n")
        print(f"Weights of Conv 1:  {m.weight.shape}")
        print(f"Weights of Conv 2:{n.weight.shape}")
        print("======= \n Test finished")

    # def test_weights(self):
    #     m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, groups=1)
    #     m_size = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
    #     input = torch.randn(10, 16, 100)  # batch, input, c_out
    #     output = m(input)
    #     m.groups = 4
    #     with torch.no_grad():
    #         m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m.groups))
    #     print(f"Weights {m.weight.shape}")
    #     m_size_2 = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
    #     output2 = m(input)

    #     print(f"Weight_Sizes before:{m_size} after:{m_size_2}")
    #     print(output.shape)
    #     print(output2.shape)
    #     # print(output)
    #     print("test ending")

    # def test_sequencedGrouping(self):
    #     """
    #         Name will be upgraded later
    #         It is a solution to the problem, that the grouping cannot be applied
    #         in random steps (so far).

    #         If we first have groups=6 and then groups=8 we can't just reshape the 'grouped=6' weights to 'groups=8' shape (based on grouped=6 weights)

    #         Though it is possible, that we have a sequenced grouping phase like 2,4,8,16,32 (which would be also conveniant for quantizied versions?)
    #         This approach will be demonstrated in that unit case
    #     """
    #     kernel_size = 3

    #     out_channels = 32
    #     in_channels = 16
    #     groups = [i for i in range(1, out_channels) if in_channels % i == 0]
    #     groups.sort(reverse=False)

    #     print(groups)

    #     m = nn.Conv1d(
    #         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups[0]
    #     )  # changes
    #     weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
    #     input = torch.randn(10, 16, 100)  # batch, input, c_out

    #     output_before = m(input)
    #     print(f"Weights {m.weight.shape}")
    #     print(output_before.shape)
    #     assert list(m.weight.shape) == weight_shape
    #     for i in groups:
    #         if(i == 1):
    #             continue
    #         m.groups = i
    #         with torch.no_grad():
    #             m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, 2))
    #             weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
    #             assert list(m.weight.shape) == weight_shape
    #         output = m(input)
    #         assert list(m.weight.shape) == weight_shape

    #     assert output_before.shape == output.shape
    #     print("test ending")

    # def test_changeGroups(self):
    #     kernel_size = 3
    #     m = nn.Conv1d(
    #         in_channels=16, out_channels=32, kernel_size=kernel_size, groups=1
    #     )  # changes
    #     weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
    #     input = torch.randn(10, 16, 100)  # batch, input, c_out
    #     output = m(input)
    #     print(f"Weights {m.weight.shape}")
    #     print(output.shape)
    #     assert list(m.weight.shape) == weight_shape
    #     m.groups = 2
    #     with torch.no_grad():
    #         m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m.groups))
    #         weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]
    #     assert list(m.weight.shape) == weight_shape

    #     output2 = m(input)

    #     # Die Mehrmalige Teilung funktioniert hier nicht.
    #     # Hier teilt er die Gewichte erneut durch 4 statt von ursprünglichen Gewichtsverhältnis
    #     # Hierbei sind zwei Wege möglich :
    #     # Gewichte durch die Differenz teilen
    #     # Ursprüngliche Gewichtsdimension merken und anhand derer die Gewichte teilen
    #     # Version eins
    #     m_groups_old = m.groups
    #     m.groups = 4
    #     # Does not work with 8 - so this is not the solution
    #     m_diff = m.groups - m_groups_old

    #     with torch.no_grad():
    #         m.weight = nn.Parameter(adjust_weights_for_grouping(m.weight, m_diff))
    #         weight_shape = [m.out_channels, m.in_channels // m.groups, kernel_size]

    #     assert list(m.weight.shape) == weight_shape
    #     output3 = m(input)

    #     print(output.shape)
    #     print(output2.shape)
    #     print(output3.shape)

    #     assert output.shape == output2.shape and output.shape == output3.shape
    #     print("test ending")


def printValues(number: int, cnn):
    print(
        f"C{number} Parameters: in_channels={cnn.in_channels}, out_channels={cnn.out_channels}, k={cnn.kernel_size}, g={cnn.groups}"
    )


if __name__ == "__main__":
    unittest.main()
