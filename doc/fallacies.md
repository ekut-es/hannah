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
# Fallacies and Pitfalls

## Changing OneCycleLR learningrate

By default the `OneCycleLR` learning rate scheduler will ignore the setting for `òptimizer.lr` and instead will
use the setting `scheduler.max_lr` to calculate each steps learning rate. To avoid problems when sweeping the the learning rate, in hannah the default setting for `scheduler.max_lr` is using the same as `optimizer.lr` but this can be overridden in project specific configuration files.
