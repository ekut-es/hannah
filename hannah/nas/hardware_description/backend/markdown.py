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
"""Translating target descriptions to markdown data sheets."""

from textwrap import dedent, wrap

from .base import DescriptionBackend


class MarkdownBackend(DescriptionBackend):
    """Generator for markdown data sheets from target devices."""

    def __init__(self, textwidth: int = 80):
        super().__init__()
        self.textwidth = textwidth

    def generate(self, device) -> str:
        """Generates a markdown description from a target device meta model."""

        text = f"# {device.name}\n\n"
        if device.description:
            desc = dedent(device.description)
            text += f"{desc}\n\n"

        text += "## Supported Operations\n\n"

        for op in device.ops:
            text += f"- {op}\n"

        return text