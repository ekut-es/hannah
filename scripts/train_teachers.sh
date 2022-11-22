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
for model in tc-res20 tc-res6 tc-res8 tc-res18; do
    hannah-train model=$model experiment_id=train_teachers scheduler=1cycle optimizer=sgd module.num_workers=8 scheduler.max_lr=1.2 trainer.max_epochs=30
    cp -r trained_models/$model/best.ckpt teachers/hannah/$model.ckpt
done
