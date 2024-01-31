from ..core.expression import Expression


# FIXME: Rename? Symbolic ...?
class SymbolicAttr(Expression):
    def __init__(self, expr, attr, key=None) -> None:
        super().__init__()
        self.expr = expr
        self.attr = attr
        self.key = key

    def get_children(self):
        return [self.expr, self.attr, self.key]

    def __getitem__(self, key):
        self.key = key
        return self

    def evaluate(self):
        attr = getattr(self.expr.evaluate(), self.attr)
        if isinstance(attr, dict) and self.key:
            attr = attr[self.key]
        elif isinstance(attr, (list, tuple)) and self.key is not None and isinstance(self.key, int):
            attr = attr[self.key]
        if hasattr(attr, 'evaluate'):
            attr = attr.evaluate()
        return attr

    def get(self, key):
        return SymbolicAttr(self, key)

    def format(self, indent=2, length=80):
        if self.key is not None:
            return f"{self.expr}({self.attr})[{self.key}]"
        else:
            return f"{self.expr}({self.attr})"


class SymbolicSequence(Expression):
    def __init__(self, expr, key) -> None:
        super().__init__()
        self.expr = expr
        self.key = key

    def __getitem__(self, key):
        self.key = key
        return self
    
    def get_children(self):
        return [self.expr, self.key]

    def evaluate(self):
        if hasattr(self.key, 'evaluate'):
            key = self.evaluate()
        else:
            key = self.key
        seq = self.expr.evaluate()[key]
        if hasattr(seq, 'evaluate'):
            seq = seq.evaluate()
        if isinstance(seq, (list, tuple)):
            evaluated_seq = []
            for s in seq:
                if hasattr(seq, 'evaluate'):
                    evaluated_seq.append(s.evaluate())
            seq = evaluated_seq

        return seq

    def format(self, indent=2, length=80):
        return f"{self.expr}({self.key})"


class Choice(Expression):
    def __init__(self, values, choice) -> None:
        super().__init__()
        self.values = values
        self.choice = choice

    def get_children(self):
        # return self.values + [self.choice]
        return [self.values, self.choice]

    def evaluate(self):
        if hasattr(self.choice, 'evaluate'):
            concrete_value = self.choice.evaluate()
        elif isinstance(self.choice, int):
            concrete_value = self.choice

        concrete_choice = self.values[concrete_value]
        if hasattr(concrete_choice, 'evaluate'):
            concrete_choice = concrete_choice.evaluate()
        return concrete_choice

    def get(self, key):
        return SymbolicAttr(self, attr=key)

    def __getitem__(self, key):
        return SymbolicSequence(expr=self, key=key)

    def format(self, indent=2, length=80):
        return f"Choice({self.values})"
