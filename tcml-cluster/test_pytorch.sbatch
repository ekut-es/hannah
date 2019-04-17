#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TestTCML

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=5:00
# the maximum time the scripts needs to run

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de
# your mail address


#Script

ls /scratch

####
#b) copy all needed data to the jobs scratch folder
####


#cp -R /common/datasets/MNIST/ /scratch/$SLURM_JOB_ID/

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

singularity exec --nv ~/speech_recognition/tcml-cluster/Speech-Recognition.simg  python3 ~/speech_recognition/tcml-cluster/test_pytorch.py

echo DONE!
