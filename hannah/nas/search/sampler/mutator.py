import numpy as np
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter


class ParameterMutator:
    def __init__(self, mutation_rate=0.05, rng=None) -> None:
        self.mutation_rate = mutation_rate
        if rng is None:
            self.rng = np.random.default_rng(seed=None)
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(seed=rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise Exception("rng should be either np.random.Generator or int (or None)")

    def mutate(self, parameters):
        mutated_keys = []
        num_mutations = int(np.ceil(self.mutation_rate * len(parameters)))
        mutation_indices = self.rng.choice(range(len(parameters)), size=num_mutations, replace=False)
        parametrization = {}
        for num, (key, param) in enumerate(parameters.items()):
            if num in mutation_indices:
                parametrization[key] = self.mutate_parameter(param)
                mutated_keys.append(key)
            else:
                parametrization[key] = param.current_value

        return parametrization, mutated_keys

    def mutate_parameter(self, parameter):
        if isinstance(parameter, CategoricalParameter):
            return self.mutate_choice(parameter)
        elif isinstance(parameter, IntScalarParameter):
            return self.mutate_int_scalar(parameter)
        elif isinstance(parameter, FloatScalarParameter):
            return self.mutate_float_scalar(parameter)

    def mutate_choice(self, parameter):
        mutations = self.get_choice_mutations()
        chosen_mutation = self.rng.choice(mutations)
        return chosen_mutation(parameter)

    def mutate_int_scalar(self, parameter):
        mutations = self.get_int_mutations()
        chosen_mutation = self.rng.choice(mutations)
        return chosen_mutation(parameter)

    def mutate_float_scalar(self, parameter):
        mutations = self.get_float_mutations()
        chosen_mutation = self.rng.choice(mutations)
        return chosen_mutation(parameter)

    # gather the relevant mutations
    def get_choice_mutations(self):
        return [self.random_choice, self.increase_choice, self.decrease_choice]

    def get_int_mutations(self):
        return [self.random_int_scalar, self.increase_int_scalar, self.decrease_int_scalar]

    def get_float_mutations(self):
        return [self.random_float_scalar]

    ####################################
    # The individual mutations
    def random_choice(self, parameter):
        return parameter.rng.choice(parameter.choices)

    def increase_choice(self, parameter):
        index = parameter.choices.index(parameter.current_value)
        if index + 1 >= len(parameter.choices):
            index = -1
        return parameter.choices[index + 1]

    def decrease_choice(self, parameter):
        index = parameter.choices.index(parameter.current_value)
        if index - 1 < 0:
            index = len(parameter.choices)
        return parameter.choices[index - 1]

    def random_int_scalar(self, parameter):
        return parameter.rng.integers(parameter.min, parameter.max+1)

    def increase_int_scalar(self, parameter):
        if parameter.current_value < parameter.max:
            return parameter.current_value + 1
        else:
            return parameter.rng.integers(parameter.min, parameter.max+1)

    def decrease_int_scalar(self, parameter):
        if parameter.current_value > parameter.min:
            return parameter.current_value - 1
        else:
            return parameter.rng.integers(parameter.min, parameter.max+1)

    def random_float_scalar(self, parameter):
        return parameter.rng.uniform(parameter.min, parameter.max)

if __name__ == '__main__':
    mutator = ParameterMutator(0.1)
    par = FloatScalarParameter(2, 5)
    print()