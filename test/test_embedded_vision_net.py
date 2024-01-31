from hannah.models.embedded_vision_net.expressions import extract_weights_recursive
from hannah.models.embedded_vision_net.models import search_space
from hannah.nas.expressions.utils import extract_parameter_from_expression
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
from hannah.nas.parameters.parametrize import set_parametrization


def test_constraint_solving():
    # FIXME: Create meaningfull tests

    # input = Tensor(name="input", shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    # space = search_space(name="evn", input=input, num_classes=10)

    # space.weights = extract_weights_recursive(space)
    # weight_params = extract_parameter_from_expression(space.weights)
    # arch.cond(And(lower, upper), weight_params)

    # solver = RandomWalkConstraintSolver()
    # # solver.build_model(space.conditions)
    # space.sample()
    # print(f"Before: Weights {space.weights.evaluate()} MACs: {space.macs.evaluate()}")
    # params = {k: p.current_value for k, p in space.parametrization().items()}
    # solver.solve(space, params)
    # solved_p = solver.get_constrained_params(None)
    # set_parametrization(solved_p, space.parametrization())
    # print(f"After: Weights {space.weights.evaluate()} MACs: {space.macs.evaluate()}")
    # print()
    pass


if __name__ == '__main__':
    test_constraint_solving()
