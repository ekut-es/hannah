from torch import nn


class DummyActivation(nn.Identity):
    """Dummy class that instantiated to mark a missing activation.

       This can be used to mark requantization of activations for convolutional layers without
       activation functions.
    """

    pass
