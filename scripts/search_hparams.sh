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

for model in tc-res4 tc-res8 conv-net-fbgemm conv-net-trax; do
    python -m hannah.train hydra/sweeper=nevergrad hydra/launcher=joblib optimizer.lr='interval(0.00001, 0.001)' optimizer.weight_decay='interval(0.000001, 0.001)' hydra.launcher.n_jobs=5 trainer.gpus=1 module.num_workers=2 early_stopping=default -m
done
