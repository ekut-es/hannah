from z3 import *
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.test.network import residual_block


class ConstraintModel:
    def __init__(self) -> None:
        self.solver = Solver()
        self.vars = {}

    def model(self):
        if self.solver.check():
            return self.solver.model()
        else:
            return None

    def check(self):
        return self.solver.check()

    def traverse(self, graph):
        queue = [graph]
        visited = [graph]

        input_dict = {}
        output_dict = {}
        enter_dict = {}

        while queue:
            graph = queue.pop(-1)

            if isinstance(graph, DataFlowGraph):
                h_in = Int(graph.id + '.h_in')
                input_dict[graph] = h_in

                for e in graph.enter:
                    enter_dict[e] = h_in
                h_out = Int(graph.id + '.h_out')

                output_dict[graph.output] = h_out

                self.solver.add(h_in > 0)
                self.solver.add(h_out > 0)

                for user in graph.users:
                    if user in input_dict and not graph == user.output:
                        self.solver.add(input_dict[user] == h_out)

                if graph in output_dict:
                    self.solver.add(h_out == output_dict[graph])

                if graph in enter_dict:
                    self.solver.add(h_in == enter_dict[graph])

                if graph.output not in visited:
                    queue.append(graph.output)
                    visited.append(graph.output)
            elif isinstance(graph, OpType):

                h_in = Int(graph.id + '.h_in')
                input_dict[graph] = h_in
                h_out = Int(graph.id + '.h_out')

                self.solver.add(h_in > 0)
                self.solver.add(h_out > 0)

                if graph.name == "Conv2d":
                    s = Int(graph.id + '.stride')
                    self.vars[graph.id + '.stride'] = s
                    k = Int(graph.id + '.kernel_size')
                    p = Int(graph.id + '.padding')
                    d = Int(graph.id + '.dilation')

                    self.solver.add(s > 0)
                    self.solver.add(s <= 2)
                    self.solver.add(k > 0)
                    self.solver.add(k <= 9)

                    self.solver.add(d == 1)
                    self.solver.add(k % 2 != 0)
                    self.solver.add(k / 2 == p)
                    # self.solver.add(p == 2)

                    self.solver.add(h_out == ((h_in + p * 2 - d * (k - 1) - 1) / s) + 1)
                else:
                    self.solver.add(h_out == h_in)

                # connect the output of this node with the input
                # of the following (user) node
                for user in graph.users:
                    if user in input_dict:
                        if hasattr(user, 'output') and graph == user.output:
                            break
                        self.solver.add(input_dict[user] == h_out)

                if graph in output_dict:
                    self.solver.add(h_out == output_dict[graph])

                if graph in enter_dict:
                    self.solver.add(h_in == enter_dict[graph])

                for o in graph.operands:
                    if o not in visited:
                        queue.append(o)
                        visited.append(o)
            elif isinstance(graph, Tensor):
                dim = Int("dim")
                if 'input' in graph.name:
                    self.solver.add(dim == int(graph['h'].size.evaluate()))

                    for user in graph.users:
                        if user in input_dict:
                            self.solver.add(input_dict[user] == dim)
                # elif 'weight' in graph.name:
                #     self.solver.add(dim == int(graph['kh'].size.evaluate()))

        print()




def test_z3():
    p = Int('p')
    s = Int('s')
    k = Int('k')
    d = Int('d')
    h_0 = Int('h_0')
    h_1 = Int('h_1')

    solver = Solver()
    solver.add(h_1 == (((h_0 + p * 2 - d * (k - 1) - 1) / s) + 1))
    solver.add(h_0 == 32)
    # solver.add(s >= 1)
    solver.add(s == 2)

    solver.add(d == 1)
    solver.add(k > 0)
    solver.add(k <= 9)
    solver.add(k % 2 != 0)
    solver.add(p == k / 2)

    pb = Int('pb')
    sb = Int('sb')
    kb = Int('kb')
    db = Int('db')
    hb_0 = Int('hb_0')
    hb_1 = Int('hb_1')

    solver.add(hb_1 == (((hb_0 + pb * 2 - db * (kb - 1) - 1) / sb) + 1))
    solver.add(hb_0 == 32)
    # solver.add(sb >= 1)
    solver.add(sb == 2)

    solver.add(db == 1)
    solver.add(kb > 0)
    solver.add(kb <= 9)
    solver.add(kb % 2 != 0)
    solver.add(pb == kb / 2)

    solver.add(hb_1 == h_1)

    print()

def test_traversal():
    input = batched_image_tensor(shape=(1,3, 32, 32), name='input')
    out = residual_block(input, stride=IntScalarParameter(1, 2), output_channel=DefaultInt(64))

    cm = ConstraintModel()
    cm.traverse(out)
    # TODO: INPUT AND OUPUTS OF GRAPHS
    print()

if __name__ == '__main__':
    # test_z3()
    test_traversal()