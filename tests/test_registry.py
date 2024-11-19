import pytest
from hannah.utils.registry import Registry

class TestClass:
    def __init__(self, x):
        self.x = x

def test_register():
    registry = Registry()
    registry.register(TestClass)
    assert 'TestClass' in registry.registered_classes

def test_instantiate():
    registry = Registry()
    registry.register(TestClass)
    instance = registry.instantiate('TestClass', 5)
    assert isinstance(instance, TestClass)
    assert instance.x == 5

def test_iter():
    registry = Registry()
    registry.register(TestClass)
    assert list(registry) == [TestClass]

def test_len():
    registry = Registry()
    registry.register(TestClass)
    assert len(registry) == 1