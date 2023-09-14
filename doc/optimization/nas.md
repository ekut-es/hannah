<!--
Copyright (c) 2022 University of Tübingen.

This file is part of hannah.
See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.

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

:warning: **Note:** We are currently in the process of restructuring the Neural Architecture Search, in particular the way search spaces are constructed. Some things might not work as usual.

# Neural architecture search

In contrast to hyperparameter optimization neural architecture search, explores new neural network hyperparameters.

A aging evolution based neural architecture search has been implemented as a hydra plugin:

   hannah-train --config-name config_unas

To launch multiple configuration jobs in parallel use joblib launcher:

    hannah-train --config-name config_unas hydra/launcher=joblib

Parametrization for neural architecture search need to be given as *YAML* configuration files at
the moment. For an example see: `speech_recognition/conf/config_unas.yaml`

## Parametrization

The Parametrization contains the following elements:

### Choice Parameter

Choice Parameters select options from a list of parameters. They are configured as a list of options in
the parameters. Example:

    conv_size: [1,3,5,7,9,11]

### Choice List Parameters

Choice List Parameters represent a variable length list of Choices. They are configured with the follwing parameters:

`min`
: Minimum length of list

`max`
: Maximum length of list+1

`choices`
: List of Choices

Example:

    min: 4
    max: 10
    choices:
      - _target_: "torch.nn.Conv2d"
        size: 3
      - _target : "torch.nn.MaxPool2d"
        size: 7

*Warning*: Mutations for intervall parameters currently always sample randomly from the range of values

### Intervall Parameters

Intervall Parameters represent a Scalar Value from an intervall of Values
They are configure with the following parameters:

`lower`
: lower bound of intervall [lower, upper[

`upper`
: upper bound of intervall [lower, upper[

`int`
: set to true to generate integers

`log`
: set to true to generate log scaled distribution

### Subset Parameter

Subset Parameters select a subset of a list of choices.

They are configured using the following parameters:

`choices`
: List of choices to sample from


`size`
: size of the subset to generate



### Partition Parameter

Partition parameters split the list of choices into a predefined number of partitions.

They are configured using the following parameters:

`choices`
: List of choices to partition

`partition`
: Number of partitions to generate
