#!/bin/bash
##
## Copyright (c) 2023 Hannah contributors.
##
## This file is part of hannah.
## See https://github.com/ekut-es/hannah for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

EXPERIMENT="baseline"
MODEL="timm_resnet18,timm_resnet50,timm_resnet152,timm_efficientnet_lite1,timm_mobilenetv3_small_100,timm_mobilenetv3_small_075,timm_mobilenetv3_large_100"

export HANNAH_DATA_FOLDER=/mnt/qb/datasets/STAGING/bringmann/datasets/

hannah-train experiment_id=$EXPERIMENT model=$MODEL hydra/launcher=ml_cloud_4gpu \
    hydra.sweep.dir='${output_dir}/${experiment_id}/' hydra.sweep.subdir='${model.name}' \
    module.num_workers=8 module.batch_size=16 trainer.gpus=4 trainer=sharded \
    -m
