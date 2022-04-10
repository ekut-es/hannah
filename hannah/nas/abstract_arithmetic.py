from abc import ABC


class AbstractArithmeticOp(ABC):
    pass


class AbstractBinaryOp(AbstractArithmeticOp):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class AbstractAdd(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractSub(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractMul(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractTruediv(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractFloordiv(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractMod(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractAnd(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)


class AbstractOr(AbstractBinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
