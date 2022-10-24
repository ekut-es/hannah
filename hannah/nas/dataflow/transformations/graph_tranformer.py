
from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph, delete_users, find_first_input, reset_scope_ids
from hannah.nas.dataflow.op_type import OpType


class GraphTransformer:
    def __init__(self, graph) -> None:
        self.graph = graph

    def transform(self, source, target, transform):
        first = find_first_input(self.graph)
        queue = [first]
        visited = [first]

        while queue:
            current = queue.pop(-1)

            if isinstance(current, DataFlowGraph) and self.match(source, current):
                self.replace_dataflow_graph(current, target, transform)

            if isinstance(current, DataFlowGraph):
                for ent in current.enter:
                    if ent not in visited:
                        queue = queue + [ent]
                        visited.append(ent)

            for user in current.users:
                if user not in visited:
                    queue = [user] + queue
                    visited.append(user)
            # if isinstance(current, DataFlowGraph):
            #     if current.output not in visited:
            #         queue = [current.output] + queue
            #         visited.append(current.output)
            # elif isinstance(current, OpType):
            #     for operand in current.operands:
            #         if operand not in visited:
            #             queue = [operand] + queue
            #             visited.append(operand)

        # self.reset_users()
        self.graph._scopes = {}
        reset_scope_ids(self.graph)
        self.graph.set_scopes()
        self.graph.collect_scopes()

    def reset_users(self):
        delete_users(self.graph)

        queue = [self.graph]
        visited = [self.graph]

        while queue:
            current = queue.pop(-1)

            if isinstance(current, DataFlowGraph):
                current.link_users()
                if current.output not in visited:
                    queue.append(current.output)
                    visited.append(current.output)
            elif isinstance(current, OpType):
                current.link_users()
                for operand in current.operands:
                    if operand not in visited:
                        queue.append(operand)
                        visited.append(operand)

    def replace_dataflow_graph(self, source, target, transform):
        # create new dfg^
        args, kwargs = transform(source, target)
        new_block = target(*args, **kwargs)  # FIXME: Correct instantiation (parameters etc)

        # new_block is automatically a user of each operand, this is not
        # necessarily correct, therefore remove here and add later if needed
        for operand in new_block.operands:
            operand.users.remove(new_block)

        # source.output.users.remove(source)
        # source.output.users.append(new_block)
        print()

        for user in source.users:
            if user.output == source:
                user.output = new_block
                del user._PARAMETERS['output']
                if is_parametrized(new_block):
                    user._PARAMETERS['output'] = new_block

            new_block.users.append(user)
            if source in user.operands:
                user.operands = list(user.operands)
                user.operands.remove(source)
                user.operands.append(new_block)
                user.operands = tuple(user.operands)

        for i, operand in enumerate(source.operands):
            if source in operand.users:
                operand.users.remove(source)
                if new_block not in operand.users:
                    operand.users.append(new_block)
                    pass
                if is_parametrized(new_block):
                    new_block._PARAMETERS[f'operand_{i}'] = operand

        parent_id = ".".join(source.id.split('.')[:-2])
        if source in self.graph._scopes[parent_id].enter:
            self.graph._scopes[parent_id].enter.remove(source)
            self.graph._scopes[parent_id].enter.append(new_block)

        del source

    def match_by_name(self, name, graph):
        if graph.name == name:
            return True
        else:
            return False

    def match_by_equivalence(self, graph_a, graph_b):
        pass

    def match(self, graph_a, graph_b):
        if isinstance(graph_a, str):
            return self.match_by_name(graph_a, graph_b)
        elif isinstance(graph_a, DataFlowGraph):
            return self.match_by_equivalence(graph_a, graph_b)
        else:
            raise Exception("Argument 0 must be either str or DataflowGraph but is {}".format(type(graph_a)))
