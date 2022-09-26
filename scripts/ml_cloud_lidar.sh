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


####
#a) Define slurm job parameters
####

#SBATCH --job-name=lidar_default

#resources:

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --gres=gpu:rtx2080ti:1
#the job can use and see 1 GPUs (8 GPUs are available in total on one node)

#SBATCH --gres-flags=enforce-binding

#SBATCH --time=720
# the maximum time the scripts needs to run (720 minutes = 12 hours)

#SBATCH --error=jobs/job_%j.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/job_%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de
# your mail address


#Script
echo "Job information"
scontrol show job $SLURM_JOB_ID

#echo "Copy training data"

#cd $tcml_wd
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_output_dir
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_data_dir

echo "Moving singularity image to local scratch"
cp /home/bringmann/cgerum05/ml_cloud.sif  $SCRATCH


echo "Moving datasets to local scratch ${SCRATCH} ${SLURM_JOB_ID}"
cp -r $WORK/datasets/KITTI_3D $SCRATCH


echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/tmp/cache
singularity run --nv -B $SCRATCH -B $WORK -H $PWD $SCRATCH/ml_cloud.sif python3 -m hannah.train experiment_id=lidar_test_1 +trainer.max_steps=10000 trainer.deterministic=false trainer.gpus=[0] dataset.DATA_PATH=$SCRATCH/KITTI_3D module.num_workers=4 output_dir=$WORK/trained_models
date

echo "DONE!"
