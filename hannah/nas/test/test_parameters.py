from collections import Counter

import numpy as np
import pytest
import scipy.stats as stats

from hannah.nas.parameters import (
    CategoricalParameter,
    FloatScalarParameter,
    IntScalarParameter,
    SubsetParameter,
)


@pytest.mark.parametrize("rng", [12312, None, np.random.default_rng(12345)])
def test_int_parameter(rng):
    parameter = IntScalarParameter(0, 10, scope=None, rng=rng)
    counter = Counter()
    for _ in range(1000):
        val = parameter.sample()
        counter[val] += 1

    for i in range(0, 10):
        assert i in counter.keys()
        prob = counter[i] / sum(counter.values())
        assert 0.05 <= prob <= 0.15


@pytest.mark.parametrize("rng", [12312, None, np.random.default_rng(12345)])
def test_int_parameter(rng):
    parameter = FloatScalarParameter(0.0, 1.0, scope=None, rng=rng)
    vals = []
    for _ in range(10000):
        val = parameter.sample()
        vals.append(val)

        assert 0.0 <= val < 1.0

    vals = np.asarray(vals)

    statistic, p_val = stats.kstest(vals, stats.uniform.cdf)
    assert p_val > 0.05
