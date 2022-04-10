from abc import ABC


class Condition(ABC):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class LTCondition(Condition):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class LECondition(Condition):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class GTCondition(Condition):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class GECondition(Condition):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class EQCondition(Condition):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class NECondition(Condition):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
