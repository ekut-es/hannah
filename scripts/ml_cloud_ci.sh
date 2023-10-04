#!/bin/bash  -l
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

cd $WORK/hannah-ci

git fetch

if [[ -z "${CI_COMMIT_SHA}" ]]; then
  export CI_COMMIT_SHA=`git rev-parse HEAD`
  echo "CI_COMMIT_SHA is not defined, use current commit (${CI_COMMIT_SHA})"
else
  git checkout $CI_COMMIT_SHA
fi

git submodule update --init --recursive

conda activate hannah-ci

srun --job-name update_env --mincpus 4 --time 01:00:00 poetry install -E vision


pushd experiments/cifar10
sbatch scripts/train_slurm.sh
