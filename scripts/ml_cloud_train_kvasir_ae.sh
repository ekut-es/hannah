#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=train_ae

#resources:

#SBATCH --cpus-per-task=4

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --gres=gpu:rtx2080ti:1
#the job can use and see 1 GPUs (8 GPUs are available in total on one node)

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
date
scontrol show job $SLURM_JOB_ID

echo "Copy training image to $SCRATCH"
date
cp /home/bringmann/cgerum05/ml_cloud.sif $SCRATCH
cp -r hannah $SCRATCH

#echo "Copy training data to $SCRATCH"
#date
#mkdir -p $SCRATCH/datasets
#cp -r $WORK/datasets/kvasir_capsule $SCRATCH/datasets

echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/cache
cd $SCRATCH
singularity run --nv --no-home  -B $SCRATCH -B $WORK -H $PWD  $SCRATCH/ml_cloud.sif python -m hannah.train -cn config_vision \
    dataset=kvasir_capsule \
    dataset.data_folder=$WORK/datasets \
    module.num_workers=4 \
    trainer.max_epochs=50 \
    module.batch_size=64 \
    experiment_id=kvasir_ae \
    output_dir=$WORK/trained_models \
    scheduler.max_lr=0.01
date

echo DONE!
