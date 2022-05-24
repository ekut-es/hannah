"""Regression test to check that summary writer is working"""

from torch.utils.tensorboard import SummaryWriter


def test_writer():
    writer = SummaryWriter(".cache")

    writer.add_scalar("test", 0.4)
    writer.close()
