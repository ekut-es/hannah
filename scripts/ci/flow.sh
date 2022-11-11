#!/usr/bin/bash -l
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

mkdir -p jobs


UPDATE_ENV_ID=$(sbatch --parsable tasks/update_env.sh)

echo "Started update env with id: ${UPDATE_ENV_ID}"


pushd ../../experiments/kws
mkdir -p jobs
KWS_ID=$(sbatch  --parsable --dependency=afterok:${UPDATE_ENV_ID} ./scripts/train_baselines_slurm.sh)
popd
