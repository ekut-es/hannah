class OpType:
    def __init__(self, name, *operands, **attributes):
        self.name = name
        self.operands = operands
        self.attributes = attributes
