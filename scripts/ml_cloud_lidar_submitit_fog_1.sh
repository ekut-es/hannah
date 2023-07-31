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
EXPERIMENT=lidar_fog_chamfer_light_fog
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
        module.augmentor.augmentations.fog.enabled=true \
        module.augmentor.augmentations.fog.metric='chamfer' \
        module.augmentor.augmentations.fog.sigma='interval(1, 100)' \
        module.augmentor.augmentations.fog.prob='interval(0, 1)' \
        hydra/sweeper=optuna \
	    hydra.sweeper.study_name=$EXPERIMENT \
	    hydra.sweeper.storage=sqlite:///$WORK/optuna_logs/${EXPERIMENT}.sqlite \
        hydra.sweeper.n_jobs=$N_JOBS \
	    hydra.sweeper.n_trials=$N_TRIALS \
	    hydra.sweeper.sampler.multivariate=true \
	    hydra/launcher=submitit_slurm \
	    hydra.launcher.timeout_min=1200 \
        hydra.launcher.gpus_per_task=1 \
        hydra.launcher.cpus_per_task=8 \
        hydra.launcher.nodes=1 \
        hydra.launcher.tasks_per_node=1 \
        hydra.launcher.partition=gpu-2080ti \
        trainer.max_epochs=80 \
        trainer.deterministic=false \
        trainer.gpus=[0] \
        dataset.DATA_PATH=/mnt/qb/datasets/STAGING/bringmann/datasets/dense_clear_original \
        dataset.VALIDATION_PATH=/mnt/qb/datasets/STAGING/bringmann/datasets/dense_light_fog \
        module.num_workers=8 \
        module.batch_size=6\
        output_dir=$WORK/trained_models \
        -m
date




echo "DONE!"
