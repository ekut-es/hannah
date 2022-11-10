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
# Torchmobile backend

A backend that runs the trained neural network through torchmobile on the local cpu.

This is an example implementation of backend that allows testing of backend integrations without installation of further packages.
It should not be used for serious optimizations, as measurements are much to noisy.


## Configuration

val_batches
: 1 (number of batches used for validation)

test_batches
: 1 (number of batches used for test)

val_frequency
: 10 (run backend every n validation epochs)
