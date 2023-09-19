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
# Pseudo Label experiments

- [ ] teste lower limit für rhode island 20 %

## Experiment 1 (Tuning)
  nur resnet 18
  nur rhode island

  - [ ] Batch Size (Maxmimum auf Resnet18+Rhode Island)

  - [ ] Learning Rate (auf Baseline)
    - [ ] One
      - [ ] max_lr

  - [ ] Softmax-Confidence Threshold (Pseudo Labeling mit SM)

  - [ ] Uncertainty-Threshold

  - [ ] Sweep %labeled



## Experiment 2 (Ablation)
### Networks
  - [ ] resnet50
  - [ ] resnet18
  - [ ] focalnet_base
  - [ ] focalnet_tiny

### Trainings
  - [ ] Baseline
  - [ ] Pseudo-Labeling
    - [ ] Softmax-Confidence
    - [ ] Softmax-Confidence + MC-Dropout Uncertainty
    - [ ] Softmax-Confidence + Fix-Match
    - [ ] Softmax-Confidence + MC-Dropout + Fix-Match

### Dataset
   - [ ] Rhode Island evtl. 2 Varianten
   - [ ] Kvasir Capsule
