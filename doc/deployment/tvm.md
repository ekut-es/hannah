<!--
Copyright (c) 2023 Hannah contributors.

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
# TVM backend

A backend that runs the trained neural network through tvm with remote target support through automate.



## Configuration

val_batches
: 1 (number of batches used for validation)

test_batches
: 1 (number of batches used for test)

val_frequency
: 10 (run backend every n validation epochs)

board
: Target board configuration

tuner
: optional tuner configuration

### Board Configuration

#### MicroTVM connected boards or simulators
#### Automate connected Remote Boards

### Tuner

Autotuning is configure using the configuration group `backend/tuner`.

There are 3 predefined tuner configurations:

1. `àutotvm`

Uses the standard configuration for autotuning on the target board, it is generallly
the recommended setting for 8-bit quantized models on CPU like devices. As it allows the
targets to use hand defined tuning templates, which generally include tvm tuning templates.

2. `auto_scheduler`

Uses the standard tvm auto_scheduler. This scheduler does not support tensorization and is therefore not recommended on targets that support tensorized instructions. This is mostly the case for tensorized instructions on the target boards.

3. `baseline`

This does not use any tuning but uses the predefined schedules directly.


All tuner configurations are parametrizable using the following tuner configurations:

### Tuner Configuration
