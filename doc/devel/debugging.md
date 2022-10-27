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
# Debugging

Hannah supports several debugging tools.

## Trainer Debugging flags:

Most of the lightning trainer [debug flags](https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html) are usable through the hydra
configs using `trainer.flag name`

The most important ones are:

### `trainer.fast_dev_run=True`

This provides a kind of unit test for the model by running one single batch of training, validation and test on the model.


### `trainer.overfit_batches=0.01`

This tests if the model can overfit a small percentage (0.01) of the dataset. A good model should be able reach almost 100 percent accuracy very quickly.


### `trainer.limit_*_batches`

This limits the amount of batches used for training. e.g.:

    hannah-train trainer.limit_train_batches=0.1 trainer.limit_val_batches=0.1 trainer.limit_test_batches=0.1

Limits the training to 10 percent of available batches for train, validation and test.
