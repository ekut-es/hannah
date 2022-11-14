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
        # currently not needed because we use a flattened graph
        # with only OpTypes and Tensors
        pass

    def process_optype(self, op: OpType):
        """Extracts the constraints based on the type of op.
        New variables are added to self.vars and the constraints
        are added to the solver.

        Parameters
        ----------
        op : OpType
        """
        if op.name == 'Conv2d':
            self.extract_conv_constraints(op)
        elif op.name == 'Add':
            self.extract_add_constraints(op)
        else:
            self.extract_passthrough_constraints(op)

    def process_tensor(self, tensor: Tensor):
        """Goes through all axis and extracts the constraints for
        the respective axis sizes

        Parameters
        ----------
        tensor : Tensor
        """
        for name, ax in tensor.tensor_type().axis.items():
            self.build_constraint_from_expression(ax.size, [])

    def extract_conv_constraints(self, op: OpType):
        input_tensor = op.operands[0].tensor_type()
        output_tensor = op.tensor_type()

        for ax_name, ax in output_tensor.axis.items():
            con = self.build_constraint_from_expression(output_tensor[ax_name].size, [input_tensor[ax_name].size])
            var = Int(f'{op.id}.{ax_name}.size')
            self.vars[str(var)] = var
            self.solver.add(var == con)

        padding = None
        kernel_size = []
        for name, var in self.vars.items():
            if 'padding' in name:
                padding = var
            elif 'kh' in name or 'kw' in name:
                kernel_size.append(var)

        for ks in kernel_size:
            self.solver.add(ks / 2 == padding)

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
        output_tensor = op.tensor_type()

        for ax_name, ax in output_tensor.axis.items():
            con = self.build_constraint_from_expression(output_tensor[ax_name].size, [input_tensor[ax_name].size])
            var = Int(f'{op.id}.{ax_name}.size')
            self.solver.add(var == con)

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
        if expr.id:
            var = Int(expr.id)
        else:
            var = Int(f"IntRange({expr.min}, {expr.max})")
            # TODO: unique scope ids for DFG parameters
        self.vars[str(var)] = var
        self.solver.add(var >= expr.min)
        self.solver.add(var <= expr.max)
        if hasattr(expr, 'step_size') and expr.step_size != 1:
            self.solver.add((var - expr.min) % expr.step_size == 0)

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
           a.id and \
           b.id and \
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
    graph = residual_block(input, stride=IntScalarParameter(1, 2), output_channel=IntScalarParameter(4, 512, 4))
    graph = flatten(graph)
    cm = ConstraintModel()
    cm.build_model(graph)
    inp = input.tensor_type()
    blck = input.users[0].tensor_type()
    print()
