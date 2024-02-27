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
""" Implementation of hannah tooling for handling hardware descriptions, and generating backends. """


import argparse
import sys

import rich

# import all devices to register them with the registry
from . import devices as _devices  # noqa: F401 # pylint: disable=unused-import
from .device import Device
from .registry import devices


def add_args(parser, known_args):
    for device in devices:
        if device.name == known_args.device:
            pass
            # device.add_args(parser, known_args)

    return parser


def list():
    for device in devices:
        print(device.name)


def export(args):
    found = False
    for device in devices:
        if device.name == args.device:
            device.export(args.output, args.backend)
            found = True
            break
    if not found:
        print(f"Device {args.device} not found.")
        print("Available devices:")
        list()
        sys.exit(1)


def describe(args):
    found = False
    import rich.markdown

    from hannah.nas.hardware_description.backend import MarkdownBackend

    for device in devices:
        if device.name == args.device:
            found = True
            device: Device = device()
            backend = MarkdownBackend()
            rich.print(rich.markdown.Markdown(backend.generate(device)))
            break
    if not found:
        print(f"Device {args.device} not found.")
        print("Available devices:")
        list()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Hardware description generator.")
    command_parsers = parser.add_subparsers(dest="command")
    command_parsers.add_parser("list", help="List all available devices.")
    export_parser = command_parsers.add_parser(
        "export", help="Generate a hardware backend."
    )
    export_parser.add_argument("device", help="Device to generate backend for.")
    export_parser.add_argument("output", help="Output directory.")
    export_parser.add_argument("--backend", help="Backend to use.", default="tvm")

    describe_parser = command_parsers.add_parser("describe", help="Describe a device.")
    describe_parser.add_argument("device", help="Device to describe.")

    known_args, _ = parser.parse_known_args()
    if known_args.command is not None:
        parser = add_args(parser, known_args)

    args = parser.parse_args()
    if args.command == "list":
        list()
    elif args.command == "export":
        export(args)
    elif args.command == "describe":
        describe(args)
