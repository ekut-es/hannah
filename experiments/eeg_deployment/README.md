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
# EEG-Model deployment for COTS platforms

Example Networks and Hardware Accelerators for EEG Seizure detection on COTS Plattforms.

Example commandline:

    hannah-train trainer.gpus=[] dataset=fake1d  backend=tvm  trainer.max_epochs=1  backend/board=stm32f429i_disc1
