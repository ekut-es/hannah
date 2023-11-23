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
# Experiment Mangement


It is common to create a new directory for each group of experiments, usually these are group around
a specific publication goal or project.

Hydra configuration options are taken from one of the following possibilities.

1. A local `config.yml` taken from the directory in which the command is run.
2. Configuration group overrides from any subdirectory of the current working directory named after a configuration group
4. Overrides from a special configuration group usually called `experiments`

for an example on experiment management have a look at `experiments/cifar10`.

It has the following directory structure:


```
.
├── augmentation
│   └── cifar_augment.yaml
├── config.yaml
├── datasets
│   └── ...
├── experiment
│   ├── sweep_lr.yaml
│   └── sweep_models.yaml
├── scripts
│   └── train_slurm.sh
└── trained_models
    ├── sweep_lr
    |   ├── 0.0001
    |   ├── ...
    └── sweep_models
        ├── multirun.yaml
        └── resnet18

```


The main configuration is found in `config.yaml`.

```yaml
defaults:
    - base_config
    - override dataset: cifar10          # Dataset configuration name
    - override features: identity        # Feature extractor configuration name (use identity for vision datasets)
    - override model: timm_resnet18      #timm_mobilenetv3_small_100      # Neural network name (for now timm_resnet50 or timm_efficientnet_lite1)
    - override scheduler: 1cycle         # learning rate scheduler config name
    - override optimizer: sgd            # Optimizer config name
    - override normalizer: null          # Feature normalizer (used for quantized neural networks)
    - override module: image_classifier  # Lightning module config for the training loop (image classifier for image classification tasks)
    - override augmentation: cifar_augment
    - _self_


monitor:
  metric: val_f1_micro
  direction: maximize

module:
  batch_size: 64

trainer:
  max_epochs: 50

scheduler:
  max_lr: 0.1
```

The configuration is composed from the standard presets in `base_config` the defaults are then using different config group presets, using the
`- override <groupname>: <presetname>` syntax. Most of the presets are taken from the package wide configuration in `hannah/conf`. The configuration for data augmentation is defined in: `augmentation/cifar_augment.yaml`:

```yaml
batch_augment:
  pipeline: null
  transforms:
    RandomVerticalFlip:
      p: 0.5
    RandomCrop:
      size: [32,32]
      padding: 4
```

This file specifies the presets for data augmentation shared among the experiments in this group of experiments.

The experiments are then defined in for example `experiment/sweep_lr.yaml` and `experiment/sweep_model.yaml`.
These experiments can be started with: `hannah-train +experiment=<experiment_name>`. Please note the **+** in front
of the commandline argument. This is needed as `experiment` is not part of default configuration hierarchy.

As an example have a look at `experiment/sweep_lr.yaml`.

```yaml
# @package _global_
experiment_id: sweep_lr
hydra:
  mode: MULTIRUN
  sweep:
    subdir: lr=${scheduler.max_lr}
  sweeper:
    params:
      scheduler.max_lr: 0.0001,0.001,0.01,0.1
```

Experiments must start with `# @package _global_` this means that overrides defined here change the global
configuration and not some values under `experiment`. The next line sets the `experiment_id` this is used
to identify the experiment and set a different subfolder for the output artifacts (normally: `trained_models/<experiment_id>`).

The final part of the configuration then configures a sweep over multiple parameters. In this case
we configure a sweep over the `max_lr` parameter of the used 1cycle learning rate scheduler, and configure the output directory
to contain a separate subdirector `lr=0.0001`, `lr=0.001` for each setting of the max_lr parameter.

The final outputs can then be found in `trained_models/sweep_lr/lr\=0.0001/` and so on.
