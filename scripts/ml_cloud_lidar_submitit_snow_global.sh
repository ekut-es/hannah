#!/bin/bash --login


#SBATCH --partition gpu-2080ti      # Run on a long cpu
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --output=jobs/%j.out      # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=jobs/%j.err       # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=BEGIN,END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de   # Email to which notifications will be sent


## Experiment
EXPERIMENT=lidar_snow_combined
RUN=_run_0
## Number of parallel jobs per node
N_JOBS=10

## Number of total trials for hyperparameter optimization
N_TRIALS=30


#Script
echo "Job information"
scontrol show job $SLURM_JOB_ID


mkdir -p $WORK/optuna_logs

conda activate hannah-lidar


date

python -m hannah.train \
        experiment_id=$EXPERIMENT$RUN \
        scheduler.max_lr=0.00004063 \
        scheduler.div_factor=21.84906 \
        optimizer.weight_decay=0.01390624 \
        module.augmentor.augmentations.snow.enabled=true \
        module.augmentor.augmentations.snow.precipitation_sigma=9.294 \
        module.augmentor.augmentations.snow.number_drops_sigma=1.505 \
        module.augmentor.augmentations.snow.prob=0.8483 \
        module.augmentor.augmentations.snow.scale=9.850 \
        module.augmentor.augmentations.snow.noise_filter_path='/mnt/qb/datasets/STAGING/bringmann/datasets/noisefilter_snow/' \
        module.augmentor.augmentations.random_translation.enabled=true \
        module.augmentor.augmentations.random_translation.sigma=2.04 \
        module.augmentor.augmentations.random_scaling.enabled=true \
        module.augmentor.augmentations.random_scaling.sigma=0.0411 \
        trainer.max_epochs=80 \
        trainer.deterministic=false \
        trainer.gpus=[0] \
        dataset.DATA_PATH=/mnt/qb/datasets/STAGING/bringmann/datasets/dense_clear_original \
        dataset.VALIDATION_PATH=/mnt/qb/datasets/STAGING/bringmann/datasets/dense_light_fog \
        module.num_workers=0 \
        module.batch_size=4\
        output_dir=$WORK/trained_models
date

echo "DONE!"
