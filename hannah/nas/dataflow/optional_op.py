class OptionalOp:
    def __init__(self, op, default):
        self.op = op
        self.default = default

    def __str__(self):
        return "optional(" + str(self.op) + str(self.default) + ")"
