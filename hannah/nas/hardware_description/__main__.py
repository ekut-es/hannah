import argparse
import sys
import rich

from . import devices as _devices
from .registry import devices


def list():
    for device in devices:
        print(device.name)   


def export(args):
    pass


def describe(args):
    device = devices.instantiate(args.device)
    rich.print(device.description)


def main():
    parser = argparse.ArgumentParser(description='Hardware description generator.')
    command_parsers = parser.add_subparsers(dest='command')
    command_parsers.add_parser('list', help='List all available devices.')
    export_parser = command_parsers.add_parser('export', help='Generate a hardware backend.')
    export_parser.add_argument('device', help='Device to generate backend for.')
    export_parser.add_argument('output', help='Output directory.')
    export_parser.add_argument('--backend', help='Backend to use.', default='tvm')
    
    describe_parser = command_parsers.add_parser('describe', help='Describe a device.')    
    describe_parser.add_argument('device', help='Device to describe.')
    
    args = parser.parse_args()
    if args.command == 'list':
        list()
    elif args.command == 'export':
        export(args)
    elif args.command == 'describe':
        describe(args)