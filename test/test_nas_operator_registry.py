from hannah.nas.search_space.modules import primitive_operators
from hannah.nas.search_space.modules.operator_registry import _OPERATORS


def test_registry():
    for name, op in _OPERATORS.items():
        print(name, op)

    assert "linear" in _OPERATORS


if __name__ == "__main__":
    test_registry()
