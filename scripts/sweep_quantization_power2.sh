#!/bin/bash
##
## Copyright (c) 2022 University of Tübingen.
##
## This file is part of hannah.
## See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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

for bits in 2 4 6; do
    python -m hannah.train trainer.max_epochs=50  model=conv-net-trax  trainer.gpus=[1]  module.num_workers=4 normalizer=fixedpoint model.qconfig.config.noise_prob=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 experiment_id=sweep_quant_noise_pot_${bits}bit model.qconfig.config.bw_w=$bits model.qconfig.config.power_of_2=true hydra/launcher=joblib hydra.launcher.n_jobs=5 -m
done
