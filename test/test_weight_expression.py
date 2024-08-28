#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from hannah.callbacks.summaries import FxMACSummaryCallback
from hannah.models.embedded_vision_net.expressions import expr_product, expr_sum
from hannah.models.embedded_vision_net.models import embedded_vision_net
from hannah.nas.expressions.choice import Choice
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import ChoiceOp, Tensor, Op
from hannah.nas.functional_operators.operators import Add, Conv2d

from hannah.nas.functional_operators.lazy import lazy
import numpy as np
import time

ADD = 1
CHOICE = 2


class FirstAddBranch:
    pass


def active_weights(node):
    queue = [node]
    visited = [node]
    s = 0
    while queue:
        n = queue.pop(0)

        if isinstance(n, Tensor) and n.id.split(".")[-1] == "weight":
            weights = np.prod([lazy(s) for s in n.shape()])
            print(f"{n.id}: {weights}")
            s += weights
        elif isinstance(n, ChoiceOp):
            idx = n.switch.evaluate()
            opt = n.options[idx]
            if opt not in visited:
                queue.append(opt)
                visited.append(opt)
        elif isinstance(n, Op):
            for operand in n.operands:
                if operand not in visited:
                    queue.append(operand)
                    visited.append(operand)
    return s


def extract_weights(graph):
    queue = [graph]
    visited = [graph]
    expr = 0

    while queue:
        node = queue.pop(0)
        if isinstance(node, ChoiceOp):
            expr = expr
        elif isinstance(node, Tensor) and node.id.split(".")[-1] == "weight":
            expr = expr + expr_product(node.shape())
        elif isinstance(node, Op):
            for operand in node.operands:
                if operand not in visited:
                    queue.append(operand)
                    visited.append(operand)


def which_scope(stack):
    for node in reversed(stack):
        if isinstance(node, Add):
            return ADD
        elif isinstance(node, ChoiceOp):
            return CHOICE
        elif isinstance(node, FirstAddBranch):
            return CHOICE


def extract_weights_recursive(node, visited, stack=[]):
    if node.id in visited:
        scope = which_scope(stack)
        if scope == ADD:
            return 0
        else:
            return visited[node.id]
    stack.append(node)
    if isinstance(node, ChoiceOp):
        exprs = []
        for o in node.options:
            w = extract_weights_recursive(o, visited, stack)
            exprs.append(w)
        c = Choice(exprs, node.switch)
        visited[node.id] = c
        stack.pop(-1)
        return c

    elif isinstance(node, Tensor) and node.id.split(".")[-1] == "weight":
        if "grouped" in node.id:
            print()
        w = expr_product(node.shape())
        visited[node.id] = w
        stack.pop(-1)
        return w
    elif isinstance(node, Op):
        exprs = []
        for i, operand in enumerate(node.operands):
            if isinstance(node, Add):
                if i == 0:
                    stack.append(FirstAddBranch())
                else:
                    stack.pop(-1)
            w = extract_weights_recursive(operand, visited, stack)
            exprs.append(w)
        s = expr_sum(exprs)
        visited[node.id] = s
        stack.pop(-1)
        return s
    else:
        stack.pop(-1)
        return 0


def test_weight_expression():
    class DummyModule:
        """Dummy module with relevant fields to demonstrate usage of
        MacSummaryCallback without the need for an actual ImageClassifierModule
        """

        def __init__(self, model, device, example_feature_array) -> None:
            self.model = model
            self.device = device
            self.example_feature_array = example_feature_array

    input = Tensor(name="input", shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    space = embedded_vision_net(name="evn", input=input, num_classes=10)

    # find a valid config
    while True:
        try:
            space.sample()
            space.check()
            break
        except Exception:
            pass

    t0 = time.perf_counter()
    model = BasicExecutor(space)
    # Determine the node execution order of the current parametrization and initialize weights
    model.initialize()

    # run a forward pass
    x = torch.randn(input.shape())
    # out = model.forward(x)

    module = DummyModule(model, "cpu", x)
    cb = FxMACSummaryCallback()
    time.perf_counter
    res = cb._do_summary(module, x)
    t1 = time.perf_counter()
    # print("Extract weights")
    # print(f"Summary time: {t1 - t0} sec")
    t0 = time.perf_counter()
    visited = {}
    nweights = extract_weights_recursive(space, visited)
    symbolically_calculated_weights = nweights.evaluate()
    t1 = time.perf_counter()
    # print(f"Expression time: {t1 - t0} sec")

    # print(f"{symbolically_calculated_weights} - {res['total_weights']}")
    assert symbolically_calculated_weights == res["total_weights"]

    # macs = space.macs.evaluate()
    # assert macs == res['total_macs']


if __name__ == "__main__":
    test_weight_expression()
