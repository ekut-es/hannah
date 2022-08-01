import numpy as np
import torch
import torch.nn as nn
import unittest

from hannah.models.ofa.submodules.elastickernelconv import ElasticConv1d
# , ElasticConvReLu1d


class OFAGroupTestCase(unittest.TestCase):

    def test_grouping(self):
        kernel_sizes = [3]
        input_length = 30
        input_channels = 8
        output_channels = 8
        batch_size = 2
        dilation_sizes = [1]
        group_sizes = [1, 2, 4, 8]

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

        loss = 1
        # warmup
        for i in range(5):
            optimizer.zero_grad()
            res = conv(input)
            loss = loss_func(res, output)
            loss.backward()
            optimizer.step()

        print("after warmup:", loss)
        group_val = {}
        for group_size in group_sizes:
            print("group_size:", group_size)
            conv.set_group_size(group_size)
            for i in range(5):
                optimizer.zero_grad()
                res = conv(input)
                loss = loss_func(res, output)
                loss.backward()
                optimizer.step()
                print("loss:", loss)

            # Validation
            validation_loss = []
            for i in range(10):
                res = conv(input)
                val_loss = loss_func(res, output)
                print("val_loss:", loss)
                validation_loss.append(val_loss.item())
            mean = np.mean(validation_loss)
            group_val[group_size] = mean

        print("Values:")
        best_pair_g = 1
        best_pair_v = 1

        for k, v in group_val.items():
            print(f"Groups {k} Accuracy {v}")
            if(v < best_pair_v):
                best_pair_v = v
                best_pair_g = k
        print(f"Best: G {best_pair_g} Accuracy {best_pair_v}")


if __name__ == '__main__':
    unittest.main()
