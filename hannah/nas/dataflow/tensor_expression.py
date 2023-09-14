class TensorExpression:
    def __init__(self, *operands, tensor_type=None, name="") -> None:
        self.operands = operands
        self._tensor_type = tensor_type
        self.users = []
        self.name = name
        self.id = name
        self._scopes = {}

    def set_id(self, id):
        self.id = id

    def tensor_type(self):
        assert self._tensor_type is not None, "Tensor Type has not been set, please run shape inference"
        return self._tensor_type

    def next_backwards(self):
        return []

    def next_forwards(self):
        return []

    def collect_scopes(self):
        queue = [self]
        visited = [self]

        while queue:
            current = queue.pop(-1)
            self._scopes[current.id] = current

            for next_node in current.next_backwards():
                if next_node not in visited:
                    queue.append(next_node)
                    visited.append(next_node)

        self._scopes = dict(sorted(self._scopes.items()))
