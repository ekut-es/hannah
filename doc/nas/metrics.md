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
# Performance Metrics in Neural Architecture Search

Currently there are two main sources of Performance Metrics used in the hannah's NAS subystem.

1. Backend generated metrics. Backends generated metrics are returned by the backend's `profile` method. Backend generated metrics are usually generated by running the neural networks, either on real target hardware or on accurate simulators. We currently do not enforce accuracy requirements on the reported metrics, but we will consider them as golden reference results for the evaluation and if necessary the training of the performance estimators, so they should be as accurate as possible.
2. Estimators can provide metrics before the neural networks have been trained. Predictors are used in presampling phases of the neural architecture search. Predictors are not and will not be used outside of neural architecture search.

There are 2 subclasses of predictors.
   - Machine Learning based predictors: These predictors provide an interface based on: `predict`, `update`, `load`, `train`
   - Analytical predictors, the interface of these methods only contains the: `predict`

The predictor interfaces are defined in `hannah.nas.performance_prediction.protcol` as python protocols.

`