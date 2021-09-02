#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ml_cloud_train

#resources:

#SBATCH --cpus-per-task=40

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --mem=64G
# the job will need 64GB of memory equally distributed on 4 cpus.

#SBATCH --gres=gpu:rtx2080ti:5
#the job can use and see 5 GPUs (8 GPUs are available in total on one node)

#SBATCH --gres-flags=enforce-binding

#SBATCH --time=4320
# the maximum time the scripts needs to run (5 minutes)

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
scontrol show job $SLURM_JOB_ID

#echo "Copy training data"

#cd $tcml_wd
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_output_dir
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_data_dir


echo "Moving datasets to local scratch ${SCRATCH} ${SLURM_JOB_ID}"

cp /home/bringmann/cgerum05/ml_cloud.simg $SCRATCH

echo "Running training with config $1"
date=
export HANNAH_CACHE_DIR=$SCRATCH/cache
singularity run --nv  --bind $PWD:/opt/hannah,$SCRATCH:/mnt $SCRATCH/ml_cloud.simg --config-name=$1 module.num_workers=4 hydra/launcher=joblib trainer.max_epochs=30  -m
date

echo DONE!
