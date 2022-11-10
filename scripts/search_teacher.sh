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


for model in tc-res4 tc-res6 tc-res8 tc-res14 tc-res16 tc-res20; do
    python -m hannah.train hydra/launcher=joblib hydra/sweeper=nevergrad hydra.launcher.n_jobs=10 \
	    optimizer=sgd scheduler=1cycle scheduler.max_lr='interval(0.1, 2.0)' model.width_multiplier=1.0,1.5,2.0,2.5 trainer.max_epochs=30 model=$model  experiment_id=search_teacher trainer.benchmark=True module.num_workers=4  -m
done
