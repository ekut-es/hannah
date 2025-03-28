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
from hannah.nas.hardware_description.backend import MarkdownBackend
from hannah.nas.hardware_description.testing import get_device


def test_simple_device():
    simple_device = get_device("simple_device")

    backend = MarkdownBackend()
    created_markdown = backend.generate(simple_device)

    print("Created Markdown:\n\n", created_markdown, sep="")


if __name__ == "__main__":
    test_simple_device()
