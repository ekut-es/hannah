from ..core.expression import Expression


class Cast(Expression):
    def __init__(self, expr, type) -> None:
        super().__init__()
        self.expr = expr
        self.type = type

    def get_children(self):
        return [self.expr, self.type]

    def evaluate(self):
        if hasattr(self.expr, 'evaluate'):
            val = self.expr.evaluate()
        else:
            val = self.expr
        return self.type(val)

    def format(self, indent=2, length=80) -> str:
        return f"{self.type}({self.expr})"


class Int(Cast):
    def __init__(self, expr) -> None:
        super().__init__(expr, type=int)
