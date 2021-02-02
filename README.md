# Deep Neural Networks for Speech Recognition

# Getting Started

## Installing dependencies

Dependencies and virtual environments are managed using [poetry](https://python-poetry.org/).

- python3.6+ and development headers
- libsndfile and development headers
- libsox and development headers
- a blas implementation and development headers

### Ubuntu 18.04+

    sudo apt update
    sudo apt -y install python3-dev libblas-dev liblapack-dev libsndfile1-dev libsox-dev

### Centos / RHEL / Scientific Linux: 7+

    sudo yum install python36 python36-devel -y
    sudo yum install portaudio-devel libsndfile1-devel libsox-devel -y


### Install poetry

    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

For alternative installation methods see:  https://python-poetry.org/docs/#installation

**Caution**: this usually install poetry to ~/.local/bin if this folder is not in your path you might need to run poetry as:

    ~/.local/bin/poetry

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

And you might need to deactivate your conda environement:

    conda deactivate

## Installing the datasets

Datasets are downloaded automatically to the datasets datafolder by default this is a subfolder of the datasets data folder.

## Training - KWS

Training is invoked by

    python -m speech_recognition.train

If available the first GPU of the system will be used by default. Selecting another GPU is possible using the argument trainer.gpus=[number]
e.g. for GPU 2 use:

    python -m speech_recognition.train trainer.gpus=[2]

Trained models are saved under `trained_models/<experiment_id>/<model_name>`.

## Training - VAD

Training of VAD is invoked by

    python -m speech_recognition.train dataset=vad model.n_labels=2

Training of VAD_Extended is invoked by

    python -m speech_recognition.train dataset=vad_extended model.n_labels=2

If available the first GPU of the system will be used by default. Selecting another GPU is possible using the argument trainer.gpus=[number]
e.g. for GPU 2 use:

    python -m speech_recognition.train dataset=vad model.n_labels=2 trainer.gpus=[2]

Trained models are saved under `trained_models/<experiment_id>/<model_name>`.

### Data Creation at the beginning

Selection of other Voice Dataset use  `dataset.variants="[UWNU, de, en, it, fr, es]" `

    python -m speech_recognition.train dataset=vad model.n_labels=2 dataset.variants="[UWNU, de, en, it, fr, es]"

Selection of other Noise Datasets use  `dataset.noise_dataset="[TUT]" `

    python -m speech_recognition.train dataset=vad model.n_labels=2 dataset.noise_dataset="[TUT]"

Selection of dataset Split use  `dataset.data_split="vad_balanced" `

    python -m speech_recognition.train dataset=vad model.n_labels=2 dataset.data_split="vad_balanced"

Create Vad_small Dataset

    python -m speech_recognition.train dataset=vad model.n_labels=2 dataset.variants="[UWNU]" dataset.noise_dataset="[TUT]" dataset.data_split="vad_balanced"

Create VAD_big Dataset

    python -m speech_recognition.train dataset=vad model.n_labels=2 dataset.variants="[UWNU, en, de, fr, es, it]" dataset.noise_dataset="[TUT, FSD50K]" dataset.data_split="vad_balanced"


# Configuration

Configurations are managed by hydra (http://hydra.cc). And follow a structured configuration.

The currently used configuration can be shown with:

   python -m speech_recognition.train  -c job

The default configurations are located under `speech_recognition/conf`.

We currently have the following configuration groups:

## backend

Configures the app to run a subset of validation and test on configured backend

Choices are: `null` (no backend, default),  `torchmobile` (translates model to backend)

## checkpoint

Choices are: `all` (default, creates checkpoints for all training batches, default) `best` (checkpoint, best 5 values)

## compress

Compression configs for distiller with activated batch norm folding:

Choices are: fp_16_16_16, fp_4_4_4, ... linear_16_16_16, linear_1_1_1, ..

Configurations have the following naming scheme: `<compression>_<bits_act>_<bits_weight>_<bits_bias>`.
All activations currently have batch norm folding activated by default. To deactivate batch norm folding, either provide a new
config file, or use `compress.bn_fold=null`.

Options:

`fold_bn` (float): use batch norm folding and freeze batch norms n % of total training

The other configuration options follow the distiller configuration format with options for quantization see: https://intellabs.github.io/distiller/schedule.html#quantization-aware-training for quantization aware training options.

## dataset

Choices are: `kws` (For Google Keyword Spotting), `snips` (For hey snips dataset), `vad` (For Voice Activity Detection)

Common configuration options for datasets are:

 - data_folder: Folder containing the data
 - cls: speech_recognition.dataset.SpeechCommandsDataset
 - dataset: # should go away

 - group_speakers_by_id: true
 - use_default_split: false
 - test_pct: Percentage of dataset used for test
 - dev_pct:  Percentage of dataset used for validation
 - train_pct: Percentage of dataset used for training
 - wanted_words: Wanted words for keyword datasets

 - input_length: "Length of the input samples"
 - samplingrate: "Sampling rate"
 - clear_download: "Remove downloaded archive after download"
 - variants: Language to use for multilanguage datasets

 - timeshift_ms: "Timeshift the input data by +- given ms"
 - extract: loudest
 - clear_download: False

 - silence_prob: % dataset samples that are silence
 - unknown_prob: % dataset samples that are unknown
 - test_snr: SNR used for test
 - train_snr_high: minimal SNR for training data
 - train_snr_low: maximal SNR for test data
 - noise_dataset: ["TUT", "FSDKaggle", "FSDnoisy"] Downlaods all the specified datasets. Use TUT + one other
 - data_split: initial split after downloading the datasets(Possibilities: "vad", "vad_speech", "vad_balanced", "getrennt")
 - downsample: samplerate DESTRUCTIVE! change the samplerate of the real files to the target samplerate.  Use better parameter samplingrate

### variants
variants for `kws`
- speech_command

variants for `snips`
- spnis

variants for `vad`
- UWNU
- Mozilla has the following language options:
    - en: Englisch
    - de: Detusch
    - fr: Französisch
    - it: Italienisch
    - es: Spanisch
    - kab: Kabylisch
    - ca: Katalanisch
    - nl: Niderländisch
    - eo: Esperanto
    - fa: Persisch
    - eu: Baskisch
    - rw: Kinyarwanda
    - ru: Russisch
    - pt: Portugiesisch
    - pl: Polnisch





## features

Feature extractor to use, choices are: melspec, mfcc, raw, sinc, spectrogram .

All feature extractors apart from `sinc` (Sinc Convolutions) currently use
torchaudio feature extractors and use the same input parameters.

Sinc Convolutions have the following parameters:

## normalizer:

Choices: `null` (default), fixedpoint

Feature Normalizer run between feature extraction and the neural network model.


## model

Neural network to train: choices are gds (Old sinc1 with sinc convolutions removed), lstm, tc-res8, branchy-tc-res8

## module

Currently only `stream_classifier` is available.

The toplevel module implementing the training and eval loops.

## optimizer

Choices are: adadelta, adam, adamax, adamw, asgd, lbfgs, rmsprop, rprop, sgd, sparse_adam

Directly instantiate the corresponding pytorch classes and take the same options.

## profiler

Choices are: `null`, advanced, simple

Run profiling for different phases.

## scheduler

Choices are: `null`, 1cycle, cosine, cosine_warm, cyclic, exponential, multistep, plateau, step

Learning rate scheduler to use for scheduling, default is null.

## trainer

Choices: default

Capsules the options to the lightning trainer. Currently it sets the following defaults:

Default options are:


 - `gpus`: 1
 - `auto_select_gpus`: True
 - `limit_train_batches`: 1.0
 - `limit_val_batches`: 1.0
 - `limit_test_batches`: 1.0
 - `max_epochs`: 80
 - `default_root_dir`: .
 - `fast_dev_run`: false
 - `overfit_batches`: 0.0
 - `benchmark`: True
 - `deterministic`: True



# Multiruns / Design space exploration

Hydra based configuration supports design space explorations for multiple configurations by default.

Hydra supports simple grid search using the parameter multirun. For example run:


    python -m speech_recognition.train --multirun  features=sinc,mfcc,melspec

For more advanced exploration techniques hydra supports sweeper plugins. The one based on nevergrad (https://facebookresearch.github.io/nevergrad/)  is installed by default.


   python -m speech_recognition.train --multirun hydra/sweeper=nevergrad  scheduler.step.lr='interval(0.0001,1.0)' scheduler.step.stepsize='interval(0.0001, 1.0)'

Sweeps over the stepsize and initial learning rate of the step learning rate scheduler.




# Showing graphical results

All experiments are logged to tensorboard: To visualize the results use:

    tensorboard --logdir trained_models

or a subdirectory of trained models if only one experiment or model is of interest.


# Neural architecture search

A aging evolution based neural architecture search has been implemented as a hydra plugin:

   python -m speech_recognition.train --config-name config_unas

To launch multiple configuration jobs in parallel use joblib launcher:

    python -m speech_recognition.train --config-name config_unas hydra/launcher=joblib

Parametrization for neural architecture search need to be given as *YAML* configuration files at
the moment. For an example see: `speech_recognition/conf/config_unas`

## Parametrization

The Parametrization contains the following elements:

### Choice Parameter

Choice Parameters select options from a list of parameters. They are configured as a list of options in
the parameters. Example:

    conv_size: [1,3,5,7,9,11]

### Choice List Parameters

Choice List Parameters represent a variable length list of Choices. They are configured with the follwing parameters:

`min` : Minimum length of list

`max` : Maximum length of list+1

`choices` : List of Choices

Example:

    min: 4
    max: 10
    choices:
      - _target_: "torch.nn.Conv2d"
        size: 3
      - _target : "torch.nn.MaxPool2d"
        size: 7

*Warning*: Mutations for intervall parameters currently always sample randomly from the range of values

### Intervall Parameters

Intervall Parameters represent a Scalar Value from an intervall of Values
They are configure with the following parameters:

`lower`: lower bound of intervall [lower, upper[
`upper`: upper bound of intervall [lower, upper[
`int`: set to true to generate integers
`log`: set to true to generate log scaled distribution
# Development

This project uses pre commit hooks for auto formatting and static code analysis.
To enable pre commit hooks run the following command in a `poetry shell`.

    pre-commit install


# TODO:

Training:

- Experiment with dilations
- Add estimation of non functional properties on algorithmic level (WIP)
