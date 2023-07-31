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

#SBATCH --gres=gpu:rtx2080ti:1
#the job can use and see 4 GPUs (8 GPUs are available in total on one node)

#SBATCH --gres-flags=enforce-binding

#SBATCH --time=4200
# the maximum time the scripts needs to run (4200 minutes = 70 hours)

#SBATCH --error=jobs/job_%j.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/job_%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de
# your mail address



## Number of parallel jobs per node
N_JOBS=2

## Number of total trials for hyperparameter optimization
N_TRIALS=2


#Script
echo "Job information"
scontrol show job $SLURM_JOB_ID

#echo "Copy training data"

#echo "Moving singularity image to local scratch"
#cp /home/bringmann/cgerum05/ml_cloud.sif  $SCRATCH


#echo "Moving datasets to local scratch ${SCRATCH} ${SLURM_JOB_ID}"
#cp -r /mnt/qb/datasets/STAGING/bringmann/datasets/dense_clear_original $SCRATCH
#cp -r /mnt/qb/datasets/STAGING/bringmann/datasets/dense_heavy_fog $SCRATCH

mkdir -p $WORK/optuna_logs

echo "Running training with config $1"
date
export HANNAH_CACHE_DIR=$SCRATCH/tmp/cache
#
#singularity run --nv -B $SCRATCH -B $WORK -H $PWD $SCRATCH/ml_cloud.sif python3 -m hannah.train \
python3 -m hannah.train \
        experiment_id=lidar_heavy_fog_test \
        trainer.max_epochs=4 \
        trainer.deterministic=false \
        dataset.DATA_PATH=/mnt/qb/datasets/STAGING/bringmann/datasets/dense_clear_original \
        dataset.VALIDATION_PATH=/mnt/qb/datasets/STAGING/bringmann/datasets/dense_heavy_fog \
        module.num_workers=2 \
        module.batch_size=6 \
        output_dir=$WORK/trained_models
date
        # module.augmentor.augmentations.random_noise.probability='interval(0, 1)' \
        # module.augmentor.augmentations.random_noise.max_points='interval(1, 10000)' \
        # hydra/sweeper=optuna \
	    # hydra.sweeper.study_name='lidar_random_noise_heavy_fog' \
	    # hydra.sweeper.storage=sqlite:///$WORK/optuna_logs/'lidar_random_noise_heavy_fog.sqlite' \
        # hydra.sweeper.n_jobs=$N_JOBS \
	    # hydra.sweeper.n_trials=$N_TRIALS \
	    # hydra.sweeper.sampler.multivariate=true \
	    # hydra/launcher=joblib \
	    # hydra.launcher.n_jobs=$N_JOBS \
        

echo "DONE!"
