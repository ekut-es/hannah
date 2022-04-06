from hannah.nas.search_space.modules.operator_registry import _OPERATORS
from hannah.nas.search_space.modules import primitive_operators


def test_registry():
    for name, op in _OPERATORS.items():
        print(name, op)

    assert "linear" in operators


if __name__ == "__main__":
    test_registry()
