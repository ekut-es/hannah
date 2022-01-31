#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ml_cloud_train

#resources:

#SBATCH --cpus-per-task=4

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --mem=8G
# the job will need 64GB of memory equally distributed on 4 cpus.

#SBATCH --gres=gpu:rtx2080ti:1
#the job can use and see 1 GPUs (8 GPUs are available in total on one node)

#SBATCH --gres-flags=enforce-binding

#SBATCH --time=4320
# the maximum time the scripts needs to run in minutes

#SBATCH --error=job_%j.err
# write the error output to job.*jobID*.err

#TSBATCH --output/home/bringmann/cgerum05/job_%j.out
#SBATCH --output=job_%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de
# your mail address


#Script
echo "Job information"
date
scontrol show job $SLURM_JOB_ID

echo "Copy training data to $SCRATCH"
date
cp /home/bringmann/cgerum05/ml_cloud.sif $SCRATCH
cp -r hannah $SCRATCH

echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/cache
cd $SCRATCH
singularity run --nv --no-home --bind $PWD  $SCRATCH/ml_cloud.sif python -m hannah.train dataset.data_folder=$SCRATCH/datasets module.num_workers=4 trainer.max_epochs=30
date
echo "Copying data folders back to work"
cp -r trained_models $WORK

echo DONE!
