#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ml_cloud_train

#resources:
#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --cpus-per-task=16
# requests 16 cpus in total

#SBATCH --gres=gpu:rtx2080ti:2
#the job can use and see 2 GPUs (8 GPUs are available in total on one node)

#SBATCH --time=4320
# the maximum time the scripts needs to run in minutes

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
date

echo "Moving singularity image to local scratch ${SCRATCH} ${SLURM_JOB_ID}"
date
cp /home/bringmann/cgerum05/ml_cloud.sif $SCRATCH
date

echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/cache
export TEDA_HOME=$PWD/external/teda/
export PYTHONPATH=$PWD/plugins/hannah-optimizer/

# Start gpu utilization logger

mkdir -p $WORK/trained_models/gpu_logs
nvidia-smi dmon -i 0,1 -s mu -d 60 -o TD -f $WORK/trained_models/gpu_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log &

singularity run --nv -B /mnt/qb/datasets/ -B $SCRATCH -B $WORK -B $PWD -H $PWD $SCRATCH/ml_cloud.sif python3 -m hannah.train +experiment=$1 dataset.data_folder=/mnt/qb/datasets/STAGING/bringmann/datasets/ module.num_workers=4 output_dir=$WORK/trained_models
date

echo DONE!
