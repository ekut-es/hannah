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


#SBATCH --job-name=lidar_default

#resources:

#SBATCH --partition=gpu-2080ti
# the slurm partition the job is queued to.
# FIXME: test if preemptable is avallable

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --gres=gpu:rtx2080ti:4
#the job can use and see 4 GPUs (8 GPUs are available in total on one node)

#SBATCH --time=720
# the maximum time the scripts needs to run (720 minutes = 12 hours)

#SBATCH --error=jobs/%j.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de
# your mail address


#Script
echo "Job information"
scontrol show job $SLURM_JOB_ID


export HANNAH_DATA_FOLDER=/mnt/qb/datasets/STAGING/bringmann/datasets/
#export HANNAH_DATA_FOLDER=$PWD/../../datasets/
export EXPERIMENT=baseline
export RESOLUTION=320
export MODEL=timm_resnet152
export SPLIT=official

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--resolution)
      RESOLUTION="$2"
      shift # past argument
      shift # past value
      ;;
    --random)
      SPLIT=random
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

GPUS="[0,1,2,3]"

# trainer=sharded

hannah-train experiment_id=${EXPERIMENT}_${SPLIT} module.num_workers=8 trainer.gpus=${GPUS} dataset.split=${SPLIT} model=${MODEL}
