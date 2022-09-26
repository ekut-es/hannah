<!--
Copyright (c) 2022 University of Tübingen.

This file is part of hannah.
See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Configuration

Configurations are managed by [hydra](http://hydra.cc). And follow a structured configuration.

The currently used configuration can be shown with:

    hannah-train  -c job

The default configuration is located under `speech_recognition/conf/`.

## Configuration Structure

There are two main locations for configuration:

- Framework configuration files are located under `hannah/conf` configuration files located in this directory should be useful for more than one project.
- Project specific configuration files are located in `configs/<project_name>` these usually contain a project specific configuration file and one or more configured experiments in the subfolder `experiment`
- make use of composition using defaults lists as much as possible

# Guideline for writing configuration files

- Consider avoiding project specific main configuration files, and try to configure your project using experiments only.
- Use composition defaults list composition as much as possible.

## Configuration Groups

We currently have the following configuration groups:

### backend

Configures the app to run a subset of validation and test on configured backend

Choices are: `null` (no backend, default),  `torchmobile` (translates model to backend)

### checkpoint

Choices are: `null` (no checkpoint), `all` (default, creates checkpoints for all training epoch) `best` (checkpoint, best 5 values, default)

### compression

Configuration for model compression.

### dataset

Choices are: `kws` (For Google Keyword Spotting), `snips` (For hey snips dataset), `vad` (For Voice Activity Detection)

Common configuration options for datasets are:

data_folder
: base folder for data (actual data will usually be stored in $data_folder/dataset_name)

cls
: speech_recognition.dataset.SpeechCommandsDataset

dataset
: name of the dataset

group_speakers_by_id
: true group the data by user id before splitting if available

use_default_split
: false if the dataset has a predefined split, use it (ignores *_pct, and group by id parameters)

test_pct
: Percentage of dataset used for test

dev_pct
:  Percentage of dataset used for validation

train_pct
: Percentage of dataset used for training

wanted_words
: Wanted words for keyword datasets

input_length
: Length of input in number of samples

samplingrate
: Sampling rate of data in Hz

clear_download
: "Remove downloaded archive after dataset has been extracted

variants
: Variant of dataset to use for multilanguage datasets

timeshift_ms
: "Timeshift the input data by +- given ms"

extract
: loudest

silence_prob
: % dataset samples that are silence

unknown_prob
: % dataset samples that are unknown

test_snr
: SNR used for test

train_snr_high
: minimal SNR for training data

train_snr_low
: maximal SNR for test data

noise_dataset
: ["TUT", "FSDKaggle", "FSDnoisy"] Downloads all the specified datasets. Use TUT + one other

data_split
: initial split after downloading the datasets(Possibilities: "vad", "vad_speech", "vad_balanced", "getrennt")

downsample
: samplerate DESTRUCTIVE! change the samplerate of the real files to the target samplerate.  Use better parameter samplingrate

#### variants
variants for `kws`
- v1, v2

variants for `snips`
- snips

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





### features

Feature extractor to use, choices are: melspec, mfcc, raw, sinc, spectrogram .

All feature extractors apart from `sinc` (Sinc Convolutions) currently use
torchaudio feature extractors and use the same input parameters.

Sinc Convolutions have the following parameters:

### normalizer

Choices: `null` (default), fixedpoint (normalize feature values to [-1.0, 1.0])

Feature Normalizer run between feature extraction and the neural network model.


### model

Neural network to train: choices are gds (Old sinc1 with sinc convolutions removed), lstm, tc-res8, branchy-tc-res8

### module

Currently only `stream_classifier` is available.

The toplevel module implementing the training and eval loops.

### optimizer

Choices are: adadelta, adam, adamax, adamw, asgd, lbfgs, rmsprop, rprop, sgd, sparse_adam

Directly instantiate the corresponding pytorch classes and take the same options.

### profiler

Choices are: `null`, advanced, simple

Run profiling for different phases.

### scheduler

Choices are: `null`, 1cycle, cosine, cosine_warm, cyclic, exponential, multistep, plateau, step

Learning rate scheduler to use for scheduling, default is null.

### trainer

Choices: default, dds

Capsules the options to the lightning trainer. Currently it sets the following defaults:

Default options are:


`gpus`
: 1

`auto_select_gpus`
: True

`limit_train_batches`
: 1.0

`limit_val_batches`
: 1.0

`limit_test_batches`
: 1.0

`max_epochs`
: 80

`default_root_dir`
: .

`fast_dev_run`
: false

`overfit_batches`
: 0.0

`benchmark`
: True

`deterministic`
: True

## Environment Variables

The default configurations interpolate the following environment variables:

`TEDA_HOME`
: Location of teda checkout for ultratrail backend


`HANNAH_CACHE_DIR`
: Location of a directory used to cache file loading in some datasets

<!--
`HANNAH_DATASETS`
: Change default location of dataset folders by default we will use subdirectory `datasets` of current working directory
-->
