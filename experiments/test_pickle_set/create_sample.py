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
import argparse
import os
import pickle

import numpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts a pickle file to a numpy file."
    )
    parser.add_argument("pickle_file", help="The pickle file to create.")
    parser.add_argument("--size", help="The number of samples in the dataset.")
    parser.add_argument(
        "--dim",
        help='The dimension of the samples, in the form of a tuple e.g. "(3, 32, 32)"',
    )
    parser.add_argument("--classes", help="The number of classes in the dataset.")

    args = parser.parse_args()

    size = int(args.size)
    dim = tuple(map(int, args.dim.strip("()").split(",")))
    classes = int(args.classes)

    with open(args.pickle_file, "wb") as f:
        pickle.dump(
            (
                numpy.random.rand(size, *dim).astype(numpy.float32),
                numpy.random.randint(0, classes, size),
            ),
            f,
        )
