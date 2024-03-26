from typing import Optional, Union
import numpy as np
from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.parameters.parameters import Parameter


def channels_per_group(n):
    channels = []
    i = n
    while i > 0:
        x = n % i
        if x == 0:
            channels.append(int(n / i))
        i -= 1
    return channels


class Groups(Parameter):
    def __init__(self,
                 in_channels: Parameter,
                 out_channels: Parameter,
                 name: Optional[str] = "",
                 rng: Optional[Union[np.random.Generator, int]] = None,):
        super().__init__(name, rng)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.current_value = self.sample()

    def get_possible_values(self):
        in_channels = lazy(self.in_channels)
        out_channels = lazy(self.out_channels)
        possible_values_in = channels_per_group(in_channels)
        possible_values_out = channels_per_group(out_channels)
        possible_values = list(set(possible_values_in).intersection(possible_values_out))
        return possible_values

    def instantiate(self):
        possible_values = self.get_possible_values()
        if self.current_value not in possible_values:
            diff = np.abs(np.array(possible_values)) - self.current_value
            self.current_value = int(possible_values[np.argmin(diff)])
        return self.current_value

    def sample(self):
        possible_values = self.get_possible_values()
        self.current_value = int(np.random.choice(possible_values))
        return self.current_value

    def check(self, value):
        possible_values = self.get_possible_values()
        if value not in possible_values:
            raise ValueError("{} channels per group not valid with {} in channels and {} out channels".format(value, self.in_channels.evaluate(), self.out_channels.evaluate()))

    def set_current(self, x):
        possible_values = self.get_possible_values()
        if x not in possible_values:
            diff = np.abs(np.array(possible_values)) - x
            self.current_value = int(possible_values[np.argmin(diff)])
        else:
            self.current_value = int(x)
