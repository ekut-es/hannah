#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ml_cloud_hyperopt

#resources:


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

## Number of parallel jobs per node
HANNAH_N_JOBS=5

## Number of total trials for hyperparameter optimization
HANNAH_N_TRIALS=1000

## Dataset Name
HANNAH_DATASET=speech_commands

## Mode Name
HANNAH_MODEL=conv-net-trax



#Script
echo "Job information"
date
scontrol show job $SLURM_JOB_ID

echo "Copy container data to $SCRATCH"
date
cp /home/bringmann/cgerum05/ml_cloud.sif $SCRATCH
cp -r hannah $SCRATCH

echo "Copy dataset to $SCRATCH"
mkdir -p $SCRATCH/datasets
cp -r $WORK/datasets/speech_commands_v0.02 $SCRATCH/datasets/speech_commands_v0.02


echo "Running training with config $1"
date

export HANNAH_CACHE_DIR=$SCRATCH/cache
cd $SCRATCH

mkdir -p $WORK/optuna_logs

singularity run --nv -B $SCRATCH -B $WORK -H $PWD  $SCRATCH/ml_cloud.sif python -m hannah.train \
	dataset.data_folder=$SCRATCH/datasets \
	module.num_workers=4 \
	trainer.max_epochs=30 \
	output_dir=$WORK/trained_models \
	experiment_id=${SLURM_JOB_NAME}_${HANNAH_DATASET} \
        hydra/sweeper=optuna \
	hydra.sweeper.study_name='${experiment_id}'_${HANNAH_MODEL} \
	hydra.sweeper.storage=sqlite:///$WORK/optuna_logs/'${experiment_id}.sqlite' \
        hydra.sweeper.n_jobs=$HANNAH_N_JOBS \
	hydra.sweeper.n_trials=$HANNAH_N_TRIALS \
	hydra.sweeper.sampler.multivariate=true \
	hydra/launcher=joblib \
	hydra.launcher.n_jobs=$HANNAH_N_JOBS \
	dataset.train_snr_high='100,200,300,400,500,600,700' \
	dataset.train_snr_low='1.0,5.0,10.0,15.0' \
	features.n_mfcc='20,30,40' \
	features.melkwargs.hop_length='128,256,512' \
	features.melkwargs.f_min='0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0' \
	features.melkwargs.f_max='4000.0,6000.0,8000.0' \
	features.melkwargs.n_mels='${features.n_mfcc}' \
	features.melkwargs.power='1.0, 2.0' \
	features.melkwargs.normalized='false,true' \
	scheduler.max_lr='tag(log,interval(1.0e-05, 0.1))' \
	optimizer=adamw \
	optimizer.weight_decay='tag(log,interval(1.0e-07, 1.0e-04))' \
	module.time_masking='int(interval(0,30))' \
	module.frequency_masking='int(interval(0,10))' \
	-m
date

echo DONE!
