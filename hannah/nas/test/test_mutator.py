from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.search.sampler.mutator import ParameterMutator


def test_int_mutations():
    param = IntScalarParameter(min=4, max=32, step_size=4)
    mutator = ParameterMutator(0.1)

    assert len(mutator.get_int_mutations(param)) == 2
    param.set_current(16)
    assert len(mutator.get_int_mutations(param)) == 3
    param.set_current(32)
    assert len(mutator.get_int_mutations(param)) == 2

    val = mutator.decrease_int_scalar(param)
    assert val == 28
    param.set_current(8)
    val = mutator.increase_int_scalar(param)
    assert val == 12

    val = mutator.random_int_scalar(param)
    assert val >= param.min
    assert val <= param.max
    assert val % param.step_size == 0

    param.set_current(32)
    val = mutator.increase_int_scalar(param)
    assert val == 32

    param.set_current(16)
    val = mutator.mutate_int_scalar(param)
    assert isinstance(val, int)
    assert val >= param.min
    assert val <= param.max
    assert val % param.step_size == 0


if __name__ == "__main__":
    test_int_mutations()
