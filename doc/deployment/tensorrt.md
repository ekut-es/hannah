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
# TensorRT Backend

For deployment on NVIDIA targets we support TensorRT backends.
Currently the TensorRT backend always compiles for the first GPU of the local system.

## Installation
The tensorrt module in the poetry shell needs to be installed seperately via pip:
```
poetry shell
pip install tensorrt
```

## Configuration

The backend supports the following configuration  options.

val_batches
: 1 (number of batches used for validation)

test_batches
: 1 (number of batches used for test)

val_frequency
: 10 (run backend every n validation epochs)

## TODO:

- [ ] remote execution support
- [ ] profiling and feedback support
