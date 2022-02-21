#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=lidar_default

#resources:

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --gres=gpu:rtx2080ti:4
#the job can use and see 4 GPUs (8 GPUs are available in total on one node)

#SBATCH --gres-flags=enforce-binding

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

#echo "Copy training data"

#cd $tcml_wd
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_output_dir
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_data_dir

echo "Moving singularity image to local scratch"
cp /home/bringmann/cgerum05/ml_cloud.sif  $SCRATCH


echo "Moving datasets to local scratch ${SCRATCH} ${SLURM_JOB_ID}"
echo "skipped"


echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/tmp/cache
singularity run --nv -B $SCRATCH -B $WORK -H $PWD $SCRATCH/ml_cloud.sif python3 -m hannah.train trainer.gpus=4 dataset.DATA_PATH=$WORK/datasets/ module.num_workers=4 output_dir=$WORK/trained_models
date

echo "DONE!"
