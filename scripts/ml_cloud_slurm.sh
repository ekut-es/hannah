#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ml_cloud_train

#resources:

#SBATCH --cpus-per-task=12

#SBATCH --partition=gpu-2080ti-preemptable
# the slurm partition the job is queued to.

#SBATCH --nodes=1
# requests that the cores are all on one node

#SBATCH --mem=12G
# the job will need 12GB of memory equally distributed on 4 cpus.

#SBATCH --gres=gpu:rtx2080ti:1
#the job can use and see 1 GPUs (8 GPUs are available in total on one node)

#SBATCH --time=5
# the maximum time the scripts needs to run (5 minutes)

#SBATCH --error=job_%j.err
# write the error output to job.*jobID*.err

#TSBATCH --output=/mnt/qb/home/bringmann/afrischknecht78/job_%j.out
#SBATCH --output=job_%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=adrian.frischknecht@uni-tuebingen.de
# your mail address


#Script
echo "Job information"
scontrol show job $SLURM_JOB_ID

#echo "Copy training data"

#cd $tcml_wd
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_output_dir
#mkdir -p /scratch/$SLURM_JOB_ID/$tcml_data_dir


#cp -R $tcml_wd/$tcml_data_dir /scratch/$SLURM_JOB_ID/

echo "Running training"
singularity exec --nv ./ml_cloud/ml_cloud.simg /bin/bash ./scripts/ml_cloud_hannah_train.sh 


#echo "Copy back output"
#cp -R /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID /home/bringmann/afrischknecht78 

echo DONE!

