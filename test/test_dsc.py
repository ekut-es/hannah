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
        new_kernel, bias = prepare_kernel_for_depthwise_separable_convolution(
            t,
            kernel=full_kernel,
            bias=None,
            in_channels=in_channel
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
        # grouping = in_channel_count
        new_kernel = prepare_kernel_for_pointwise_convolution(
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


def printValues(number: int, cnn):
    print(
        f"C{number} Parameters: in_channels={cnn.in_channels}, out_channels={cnn.out_channels}, k={cnn.kernel_size}, g={cnn.groups}"
    )


if __name__ == "__main__":
    unittest.main()
