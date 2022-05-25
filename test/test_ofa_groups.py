import numpy as np
import torch
import torch.nn as nn

from hannah.models.ofa.submodules.elastickernelconv import ElasticConv1d


def test_elastic_conv1d_groups():
    kernel_sizes = [3]
    input_length = 30
    input_channels = 32
    output_channels = 8
    batch_size = 2
    dilation_sizes = [1]
    group_sizes = [1, 2]
    group_sizes = [4]
    group_sizes = [2,1]

    # FRAGE: wird group size während einem step verändert
    # check calls of set_group_size

    input = torch.ones((batch_size, input_channels, input_length))
    output = torch.zeros((batch_size, output_channels, input_length))

    conv = ElasticConv1d(
        input_channels, output_channels, kernel_sizes, dilation_sizes=dilation_sizes, groups=group_sizes
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)

    res = conv(input)
    orig_loss = loss_func(res, output)
    print("orig_loss:", orig_loss)

    assert res.shape == output.shape

    for i in range(5):
        optimizer.zero_grad()
        res = conv(input)
        loss = loss_func(res, output)
        loss.backward()
        optimizer.step()

    # Sample convolution size
    random_sample_groupsizes = []
    for i in range(20):
        random_sample_groupsizes.append(np.random.choice(group_sizes))

    random_sample_groupsizes.sort(reverse=False)

    for i in range(20):
        group_size = random_sample_groupsizes[i]
        print("Sampled Group Size:", group_size)
        conv.set_group_size(group_size, False)
        optimizer.zero_grad()
        res = conv(input)
        loss = loss_func(res, output)
        loss.backward()
        optimizer.step()

    for group_size in group_sizes:
        conv.set_group_size(group_size, True)
        res = conv(input)
        loss = loss_func(res, output)
        print("group_size:", group_size, "loss:", loss)

        assert loss < orig_loss


def test_elastic_conv1d_groups_one_by_one():
    kernel_sizes = [3]
    input_length = 30
    input_channels = 8
    output_channels = 8
    batch_size = 2
    dilation_sizes = [1]
    group_sizes = [1, 2, 4, 8]

    # FRAGE: wird group size während einem step verändert
    # check calls of set_group_size

    input = torch.ones((batch_size, input_channels, input_length))
    output = torch.zeros((batch_size, output_channels, input_length))

    conv = ElasticConv1d(
        input_channels, output_channels, kernel_sizes, dilation_sizes=dilation_sizes, groups=group_sizes
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)

    res = conv(input)
    orig_loss = loss_func(res, output)
    print("orig_loss:", orig_loss)

    assert res.shape == output.shape

    for i in range(5):
        optimizer.zero_grad()
        res = conv(input)
        loss = loss_func(res, output)
        loss.backward()
        optimizer.step()

    # Sample convolution size
    random_sample_groupsizes = []
    for i in range(20):
        random_sample_groupsizes.append(np.random.choice(group_sizes))

    random_sample_groupsizes.sort(reverse=False)

    for i in range(20):
        group_size = np.random.choice(group_sizes)
        print("Sampled Group Size:", group_size)
        conv.set_group_size(group_size, False)
        optimizer.zero_grad()
        conv = ElasticConv1d(
            input_channels, output_channels, kernel_sizes, dilation_sizes=dilation_sizes, groups=[group_size]
        )
        res = conv(input)
        loss = loss_func(res, output)
        loss.backward()
        optimizer.step()
        print("loss_random:", loss)

    for group_size in group_sizes:
        optimizer.zero_grad()
        conv.set_group_size(group_size, True)
        conv = ElasticConv1d(
            input_channels, output_channels, kernel_sizes, dilation_sizes=dilation_sizes, groups=[group_size]
        )
        res = conv(input)
        loss = loss_func(res, output)
        print("group_size:", group_size)
        print("loss:", loss)



if __name__ == "__main__":
    test_elastic_conv1d_groups_one_by_one()
