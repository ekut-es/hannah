class TransformRegistry:
    def __init__(self):
        self.transforms = {}

    def register(self, cls):
        self.transforms[cls.__name__] = cls

    def instantiate(self, name, **params):
        return self.transforms[name](**params)


registry = TransformRegistry()
