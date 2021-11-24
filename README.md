# HANNAH - Hardware Acccelerator and Neural network searcH

# Getting Started

## Installing dependencies

Dependencies and virtual environments are managed using [poetry](https://python-poetry.org/).

- python (>=3.7.1 <3.10) and development headers
- libsndfile and development headers
- libsox and development headers
- a blas implementation and development headers
- git-lfs for management of checkpoints

### Ubuntu 18.04+

    sudo apt update
    sudo apt -y install python3-dev libblas-dev liblapack-dev libsndfile1-dev libsox-dev git-lfs

### Centos / RHEL / Scientific Linux: 7+

    sudo yum install python36 python36-devel -y
    sudo yum install portaudio-devel libsndfile1-devel libsox-devel -y

### Mac OS
Python 3.9 should be installed. Possible way are Homebrew

    brew install python@3.9

### Install poetry

    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

For alternative installation methods see:  https://python-poetry.org/docs/#installation

**Caution**: this usually install poetry to ~/.local/bin if this folder is not in your path you might need to run poetry as:

    ~/.local/bin/poetry

#### Mac OS
Install poetry from pip

    pip3 install poetry

## Software installation

In the root directory of the project run:

    git submodule update --init --recursive
    poetry run pip install --upgrade pip
    poetry install

This creates a virtual environment under ~/.cache/pypoetry/virtualenvs.

The environment can be activated using:

    poetry shell

### Installation Tips

1.) venv location

poetry installs the dependencies to a virtual environment in ~/.cache/pypoetry/virtualenvs

You can change the location of this directory using:

    poetry config virtualenvs.path  <desired virtual environment path>

Or move it to a subdirectory of the project directory using:

    poetry config virtualenvs.in-project true

2.) On lucille

Put the following in `.config/pip/pip.conf` until

    [global]
    timeout = 60
    extra-index-url = https://atreus.informatik.uni-tuebingen.de/~gerum/dist/

And install pytorch manually in your poetry env.

    poetry shell
    pip install torch==1.8.1 torchvision torchaudio

And you might need to deactivate your conda environement:

    conda deactivate



## Installing the datasets

Datasets are downloaded automatically to the datasets data folder by default this is a subfolder of the dataset's data folder.

For the VAD Dataset the following Flag is needed to Download/Override the Dataset

    dataset.override=True

## Training - Keyword Spotting

Training is invoked by

    hannah-train

If available the first GPU of the system will be used by default. Selecting another GPU is possible using the argument trainer.`gpus=[number]`
e.g. for GPU 2 use:

    hannah-train trainer.gpus=[2]

Trained models are saved under `trained_models/<experiment_id>/<model_name>`.

## Training - VAD

Training of VAD is invoked by

    hannah-train dataset=vad model.n_labels=2

Training of VAD_Extended is invoked by

    hannah-train dataset=vad_extended model.n_labels=2

### VAD dataset variants

Selection of other Voice Dataset use  `dataset.variants="[UWNU, de, en, it, fr, es]" `

    hannah-train dataset=vad model.n_labels=2 dataset.variants="[UWNU, de, en, it, fr, es]"

Selection of other Noise Datasets use  `dataset.noise_dataset="[TUT]" `

    hannah-train dataset=vad model.n_labels=2 dataset.noise_dataset="[TUT]"

Selection of dataset Split use  `dataset.data_split="vad_balanced" `

    hannah-train dataset=vad model.n_labels=2 dataset.data_split="vad_balanced"

Create Vad_small Dataset

    hannah-train dataset=vad model.n_labels=2 dataset.variants="[UWNU]" dataset.noise_dataset="[TUT]" dataset.data_split="vad_balanced"

Create VAD_big Dataset

    hannah-train dataset=vad model.n_labels=2 dataset.variants="[UWNU, en, de, fr, es, it]" dataset.noise_dataset="[TUT, FSD50K]" dataset.data_split="vad_balanced"


## Training - PAMAP2

Training of PAMAP2 human activity detection dataset is invoked by:

    hannah-train -cn config_activity

# Parallel Launchers

To launch multiple optimizations in parallel you can use a hydra launcher

Submitit launcher is installed by default:

   hannah-train --multirun hydra/sweeper=nevergrad hydra/launcher=joblib optimizer.lr='interval(0.0001,0.1)' optimizer.weight_decay='interval(0, 0.1)' hydra.launcher.n_jobs=5

Launches optimizer hyerparameter optimization with 5 parallel jobs.

# Early stopping

To stop training early when a validation metric does not improve, you can use lighning's early stopping callback:

    hannah-train early_stopping=default


# Showing graphical results

All experiments are logged to tensorboard: To visualize the results use:

    tensorboard --logdir trained_models

or a subdirectory of trained models if only one experiment or model is of interest.

# Development

This project uses precommit hooks for auto formatting and static code analysis.
To enable precommit hooks run the following command in a `poetry shell`.

     pre-commit install

Try to follow (pep8)[https://pep8.org/#naming-conventions] and the rest of pep8 to the
best of your abilities.
