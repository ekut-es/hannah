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


#SBATCH --job-name=run-random_nas

#resources:

#SBATCH --partition=gpu-2080ti
# the slurm partition the job is queued to.
# FIXME: test if preemptable is avallable

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --gres=gpu:rtx2080ti:8
#the job can use and see 4 GPUs (8 GPUs are available in total on one node)

#SBATCH --time=4320
# the maximum time the scripts needs to run (720 minutes = 12 hours)

#SBATCH --error=jobs/%j.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=moritz.reiber@uni-tuebingen.de
# your mail address


#Script
echo "Job information"
scontrol show job $SLURM_JOB_ID



# export HANNAH_DATA_FOLDER=/mnt/qb/datasets/STAGING/bringmann/datasets/
conda activate hannah


hannah-train trainer.gpus=8 experiment=ae_nas_cifar10_v2 model=embedded_vision_net dataset=cifar10 model.num_classes=10 nas.n_jobs=8 fx_mac_summary=True ~normalizer
