from z3 import Solver, Int, Or
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph, flatten
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.ops import batched_image_tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter, Parameter
from hannah.nas.test.network import residual_block
from hannah.nas.expressions.placeholder import Categorical, DefaultInt, IntRange, Placeholder
from hannah.nas.expressions.op import UnaryOp, BinaryOp
from hannah.nas.expressions.arithmetic import Add, Floor, Truediv, Mul, Sub


class ConstraintModel:
    def __init__(self) -> None:
        self.solver = Solver()
        self.vars = {}

        self.input_dict = {}
        self.output_dict = {}
        self.enter_dict = {}

    def build_model(self, graph):
        queue = [graph]
        visited = [graph]

        while queue:
            graph = queue.pop(-1)

            if isinstance(graph, DataFlowGraph):
                self.process_dataflow(graph)

                if graph.output not in visited:
                    queue.append(graph.output)
                    visited.append(graph.output)

            elif isinstance(graph, OpType):
                self.process_optype(graph)

                for o in graph.operands:
                    if o not in visited:
                        queue.append(o)
                        visited.append(o)

            elif isinstance(graph, Tensor):
                self.process_tensor(graph)

    def process_dataflow(self, graph):
        for operand in graph.operands:
            input_tensor = operand.tensor_type()
            self.input_dict[graph] = {}
            self.output_dict[graph.output] = {}
            for e in graph.enter:
                self.enter_dict[e] = {}

            for ax_name, ax in input_tensor.axis.items():
                ax_in = Int(f'{graph.id}.ax_{ax_name}.in')
                ax_out = Int(f'{graph.id}.ax_{ax_name}.out')

                self.vars[str(ax_in)] = ax_in
                self.vars[str(ax_out)] = ax_out

                self.input_dict[graph][ax_name] = ax_in
                self.output_dict[graph.output][ax_name] = ax_out
                for e in graph.enter:
                    self.enter_dict[e][ax_name] = ax_in

                self.link_output_to_next_input(graph, ax_name, ax_out)
                self.passthrough_output(graph, ax_name, ax_out)
                self.passthrough_enter(graph, ax_name, ax_in)

        print()

    def process_optype(self, op):
        if op.name == 'Conv2d':
            self.extract_conv_constraints(op)
        elif op.name == 'Add':
            self.extract_add_constraints(op)
        else:
            self.extract_passthrough_constraints(op)
        # output_tensor = op.tensor_type()
        # for operand in op.operands:
        #     input_tensor = operand.tensor_type()
        #     self.input_dict[op] = {}
        #     for ax_name, ax in input_tensor.axis.items():
        #         ax_in = Int(f'{op.id}.ax_{ax_name}.in')
        #         ax_out = Int(f'{op.id}.ax_{ax_name}.out')
        #         self.input_dict[op][ax_name] = ax_in

    def process_tensor(self, tensor):
        for name, ax in tensor.tensor_type().axis.items():
            self.build_constraint_from_expression(ax.size, [])

        print()

    def link_output_to_next_input(self, graph, ax_name, out_var):
        for user in graph.users:
            if user in self.input_dict and not graph == user.output:
                self.solver.add(self.input_dict[user][ax_name] == out_var)

    def passthrough_output(self, graph, ax_name, out_var):
        if graph in self.output_dict:
            self.solver.add(out_var == self.output_dict[graph][ax_name])

    def passthrough_enter(self, graph, ax_name, in_var):
        if graph in self.enter_dict:
            self.solver.add(in_var == self.enter_dict[graph][ax_name])

    def extract_conv_constraints(self, op):
        # self.solver.add(kernel_size / 2 == padding) TODO; Padding constraint
        input_tensor = op.operands[0].tensor_type()
        output_tensor = op.tensor_type()

        n_con = self.build_constraint_from_expression(output_tensor['n'].size, [input_tensor['n'].size])
        c_con = self.build_constraint_from_expression(output_tensor['c'].size, [input_tensor['c'].size])
        h_con = self.build_constraint_from_expression(output_tensor['h'].size, [input_tensor['h'].size])
        w_con = self.build_constraint_from_expression(output_tensor['w'].size, [input_tensor['w'].size])

        padding = None
        kernel_size = []
        for name, var in self.vars.items():
            if 'padding' in name:
                padding = var
            elif 'kh' in name or 'kw' in name:
                kernel_size.append(var)

        for ks in kernel_size:
            self.solver.add(ks / 2 == padding)

        n_var = Int(op.id + '.n.size')
        c_var = Int(op.id + '.c.size')
        h_var = Int(op.id + '.h.size')
        w_var = Int(op.id + '.w.size')

        self.solver.add(n_var == n_con)
        self.solver.add(c_var == c_con)
        self.solver.add(h_var == h_con)
        self.solver.add(w_var == w_con)

    def extract_add_constraints(self, op):
        output_tensor = op.tensor_type()

        for name, ax in output_tensor.axis.items():
            ax_out = Int(f'{op.id}.{name}.size')

            for operand in op.operands:
                input_tensor = operand.tensor_type()
                for in_name, in_ax in input_tensor.axis.items():
                    ax_in = Int(f'{operand.id}.{name}.size')
                    self.solver.add(ax_in == ax_out)

    def extract_passthrough_constraints(self, op):
        input_tensor = op.operands[0].tensor_type()

        for ax_name, ax in input_tensor.axis.items():
            ax_in = Int(f'{op.operands[0].id}.{ax_name}.size')
            ax_out = Int(f'{op.id}.{ax_name}.size')
            self.solver.add(ax_in == ax_out)

    def extract_parameter(self, expr):
        if isinstance(expr, (IntScalarParameter, IntRange)):
            return self.extract_int_range(expr)
        elif isinstance(expr, (CategoricalParameter, Categorical)):
            return self.extract_categorical(expr)
        elif isinstance(expr, DefaultInt):
            return self.extract_defaultint(expr)
        elif isinstance(expr, int):
            var = Int(expr.id)
            self.solver.add(var == expr)
            return var

    def extract_int_range(self, expr):
        var = Int(expr.id)
        self.vars[expr.id] = var
        self.solver.add(var >= expr.min)
        self.solver.add(var <= expr.max)
        if hasattr(expr, 'step_size') and expr.step_size != 1:
            self.solver.add((var - expr.min) % expr.step_size != 0)

        return var

    def extract_categorical(self, expr):
        var = Int(expr.id)
        self.vars[expr.id] = var
        cons = []
        for val in expr.choices:
            cons.append(var == val)
        self.solver.add(Or(cons))
        return var

    def extract_defaultint(self, expr):
        if expr.id:
            var = Int(expr.id)
        else:
            var = Int(f'DefaultInt({expr.value})')
        self.vars[str(var)] = var
        self.solver.add(var == expr.value)
        return var

    def build_constraint_from_expression(self, expr, inputs):
        for inp in inputs:
            if check_for_id(expr, inp):
                in_var = Int(inp.id)
                self.vars[inp.id] = in_var
                return in_var
        if isinstance(expr, Parameter):
            var = self.extract_parameter(expr)
            self.vars[str(var)] = var
            return var
        elif isinstance(expr, Placeholder):
            var = self.extract_parameter(expr)
            return var
        elif isinstance(expr, Add):
            lhs = self.build_constraint_from_expression(expr.lhs, inputs)
            rhs = self.build_constraint_from_expression(expr.rhs, inputs)
            con = lhs + rhs
            return con
        elif isinstance(expr, Truediv):
            lhs = self.build_constraint_from_expression(expr.lhs, inputs)
            rhs = self.build_constraint_from_expression(expr.rhs, inputs)
            con = lhs / rhs
            return con
        elif isinstance(expr, Mul):
            lhs = self.build_constraint_from_expression(expr.lhs, inputs)
            rhs = self.build_constraint_from_expression(expr.rhs, inputs)
            con = lhs * rhs
            return con
        elif isinstance(expr, Sub):
            lhs = self.build_constraint_from_expression(expr.lhs, inputs)
            rhs = self.build_constraint_from_expression(expr.rhs, inputs)
            con = lhs - rhs
            return con
        elif isinstance(expr, Floor):
            con = self.build_constraint_from_expression(expr.operand, inputs)
            return con
        elif isinstance(expr, int):
            var = Int(f"Literal({expr})")
            self.solver.add(var == expr)
            return var


def check_for_id(a, b):
    return hasattr(a, 'id') and \
           hasattr(b, 'id') and \
           a.id == b.id


def find_operand_in_expression(operand, expr):
    queue = [expr]
    visited = [expr]

    while queue:
        current = queue.pop(-1)
        if isinstance(current, UnaryOp):
            print("Check Unary")
            if check_for_id(current.operand, operand):
                print('found')
            else:
                queue.append(current.operand)
                visited.append(current.operand)
        elif isinstance(current, BinaryOp):
            print("Check Binary")
            if check_for_id(operand, current.lhs):
                print("Found lhs")
            elif check_for_id(operand, current.rhs):
                print('Found rhs')
            else:
                queue.append(current.lhs)
                queue.append(current.rhs)
                visited.append(current.lhs)
                visited.append(current.rhs)


if __name__ == '__main__':
    cm = ConstraintModel()
    input = batched_image_tensor(shape=(1, 3, 32, 32), dtype=CategoricalParameter(choices=['int6', 'int8']), name='input')
    graph = residual_block(input, stride=IntScalarParameter(1, 2), output_channel=DefaultInt(64))
    graph = flatten(graph)
    cm = ConstraintModel()
    cm.build_model(graph)
    inp = input.tensor_type()
    blck = input.users[0].tensor_type()
    print()
