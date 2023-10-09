from hannah.nas.core.expression import Expression
from hannah.nas.core.parametrized import is_parametrized


def extract_parameter_from_expression(expression):
    assert isinstance(expression, Expression)
    queue = expression.get_children()
    params = []
    visited = expression.get_children()
    while queue:
        current = queue.pop(0)
        if is_parametrized(current):
            params.append(current)
        elif isinstance(current, Expression):
            children = current.get_children()
            for c in children:
                if not any([c is v for v in visited]):  # Hack because EQCondition messes with classical "if x in list" syntax
                    queue.append(c)
                    visited.append(c)
    return params
