#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=kws_default

#resources:

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --gres=gpu:rtx2080ti:1
#the job can use and see 5 GPUs (8 GPUs are available in total on one node)

#SBATCH --time=300
# the maximum time the scripts needs to run (300 minutes = 5 hours)

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

echo "Copy training data"
mkdir -p $SCRATCH/datasets
cp -r $WORK/datasets/speech_commands_v0.02 $SCRATCH/datasets

echo "Moving singularity image to local scratch"
cp §WORK/ml_cloud.sif  $SCRATCH



echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/tmp/cache
singularity run --nv -B $SCRATCH -B $WORK -H $PWD $SCRATCH/ml_cloud.sif python3 -m hannah.train dataset.data_folder=$SCRATCH/datasets module.num_workers=8 output_dir=$WORK/trained_models
date

echo "DONE!"
