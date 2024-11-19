#
# Copyright (c) 2023 Hannah contributors.
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

import os
import sys

from pytest import importorskip

importorskip("tvm")

from tvm import relay
from tvm.relay import transform
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.testing.resnet import get_workload

from hannah.nas.functional_operators.operators import Conv2d, Tensor
from hannah.nas.hardware_description.backend import TVMBackend
from hannah.nas.hardware_description.testing import get_device
from hannah.nas.parameters import CategoricalParameter

"""

Unfied device desriptions for neural architecture search (HANNAH), and compilation (TVM, equality saturation)

"""


def test_simple_device():
    simple_device = get_device("simple_device", relu=False)

    backend = TVMBackend()
    tvm_target = backend.generate(simple_device)

    sys.path.append(".")

    os.makedirs("temp", exist_ok=True)
    with open("temp/pattern_table.py", "w") as f:
        f.write(tvm_target)

    import temp.pattern_table as pt

    table = pt.pattern_table()
    print(table)

    print(get_pattern_table("simple_device"))

    mod, params = get_workload()

    mod = transform.MergeComposite(table)(mod)
    print(mod)

    # mod = transform.AnnotateTarget(simple_device.name, False)(mod)
    # mod = transform.MergeCompilerRegions()(mod)
    # mod = transform.PartitionGraph()(mod)


if __name__ == "__main__":
    test_simple_device()
