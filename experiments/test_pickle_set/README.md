<!--
Copyright (c) 2024 Hannah contributors.

This file is part of hannah.
See https://github.com/ekut-es/hannah for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Test Pickle dataset

A simple test implementation for pickled datasets.

The pickled datasets are expected to contain a tuple of numpy arrays.

The first array contains the (preprocessed) input data, the second array contains the target class ids as int32 values.

## Creating Test Data

The following creates test, val and train datasets with 400, 400 and 4000 samples respectively.
The data is randomly initialized, and the classes are also randomly attached to a number of 2.

    python create_sample.py  --size 400 --dim='(20,17)' --classes=2 test.pkl
    python create_sample.py  --size 400 --dim='(20,17)' --classes=2 val.pkl
    python create_sample.py  --size 4000 --dim='(20,17)' --classes=2 train.pkl

## Training

This then runs a training on tc-res8

    hannah-train
