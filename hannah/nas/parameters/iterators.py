class RangeIterator:
    def __init__(self, parameter, instance=False) -> None:
        self.parameter = parameter
        self.instance = instance
        self.counter = 0 - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter < (self.parameter.current_value if self.instance else self.parameter.max):
            return self.counter
        raise StopIteration
