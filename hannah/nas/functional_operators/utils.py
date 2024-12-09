from hannah.nas.functional_operators.op import ChoiceOp, Tensor
from hannah.models.embedded_vision_net.models import embedded_vision_net
from hannah.nas.parameters.parameters import Parameter
from hannah.nas.core.expression import Expression


def get_active_parameters(space, parametrization=None):
    if parametrization is None:
        parametrization = space.parametrization()

    queue = [space]
    visited = [space.id]
    active_params = {}

    while queue:
        node = queue.pop(0)
        for k, p in node._PARAMETERS.items():
            if isinstance(p, Parameter):
                active_params[p.id] = parametrization[p.id]
        for operand in node.operands:
            while isinstance(operand, ChoiceOp):
                for k, p in operand._PARAMETERS.items():
                    if isinstance(p, Parameter):
                        active_params[p.id] = parametrization[p.id]
                active_op_index = operand.switch.evaluate()
                operand = operand.operands[active_op_index]
            if operand.id not in visited:
                queue.append(operand)
                visited.append(operand.id)
    return active_params
