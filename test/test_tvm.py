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
import subprocess
from pathlib import Path

import pytest

tvm = pytest.importorskip("tvm")

topdir = Path(__file__).parent.absolute() / ".."


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,board,tuner",
    [
        ("conv-net-trax", "local_cpu", "baseline"),
    ],
)
def test_tvm_integration(model, board, tuner):
    command_line = f"python -m hannah.tools.train trainer.fast_dev_run=True model={model} backend=tvm backend/board={board} backend/tuner={tuner}"
    print(command_line)
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
