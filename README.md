# Deep Neural Networks for Speech Recognition

# Getting Started

## Installing dependencies

Dependencies and virtual environments are managed using [poetry](https://python-poetry.org/).

- python3.6 and development headers
- portaudio and development headers
- freeglut and development headers
- a blas implementation and development headers

### Ubuntu 18.04+

    sudo apt update
    sudo apt -y install python3-dev freeglut3-dev portaudio19-dev libblas-dev liblapack-dev

### Centos / RHEL / Scientific Linux: 7+

    sudo yum install python36 python36-devel -y || true
    sudo yum install freeglut-devel -y
    sudo yum install portaudio-devel -y


### Install poetry

    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

For alternative installation methods see:  https://python-poetry.org/docs/#installation

**Caution**: this usually install poetry to ~/.local/bin it this folder is not in your path you might need to run poetry as:

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

Installing dataset for KWS:

    cd datasets
	./get_datasets.sh

Installing dataset for VAD:

    cd datasets
    ./get_datasets_vad.sh

## Training - KWS

Training on CPU can be invoked by:

    python -m speech_recognition.train  --no_cuda  --model ekut-raw-cnn3-relu

Training on 1st GPU can be invoked by:

    python -m speech_recognition.train  --gpu_no 0  --model ekut-raw-cnn3-relu

Trained models are saved under trained_models/model_name .

## Evaluation

To run only the evalution of a model use:

    python -m speech_recognition.train --no_cuda 0 --model ekut-raw-cnn3-relu --batch_size 256 --input_file trained_models/ekut-raw-cnn3-relu/model.pt --type eval

## Training - VAD

Training for the simple-vad, bottleneck-vad and small-vad can be invoked by:

    python -m speech_recognition.train --model simple-vad      --dataset vad --data_folder datasets/vad_data_balanced --n-labels 2 --lr 0.001 --silence_prob 0 --early_stopping 3 --input_length 6360 --stride 2 --conv1_size 5 --conv1_features 40 --conv2_size 5 --conv2_features 20 --conv3_size 5 --conv3_features 10 --fc_size 40
    python -m speech_recognition.train --model bottleneck-vad  --dataset vad --data_folder datasets/vad_data_balanced --n-labels 2 --lr 0.001 --n-mfcc 40 --silence_prob 0 --early_stopping 3 --input_length 6360 --stride 2 --conv1_size 5 --conv1_features 40 --conv2_size 5 --conv2_features 20 --conv3_size 5 --conv3_features 10 --fc_size 250
    python -m speech_recognition.train --model small-vad       --dataset vad --data_folder datasets/vad_data_balanced --n-labels 2

# Showing graphical results

To show visual results as a multi-axis plot, execute the following command in speech recognition's root path:

    python -m visualize.visualize --model <model> --experiment_id <experiment_id> (--top_n_accuracy <top_n_accuracy>)

Please note, that an axis, that has equal values for all variations, is dropped from the graph for the sake of clarity.

You have to have a browser installed on your system to see the results. If you have a non-graphical system, please copy the experiment folder from `<speech_recognition_root>/trained_models/<experiment_id>` to the `trained_models` folder of another machine with graphical support.

# TODO:

Training:

- Implement Wavenet
- Experiment with dilations
- Add design space exploration tool (WIP)
- Add estimation of non functional properties on algorithmic level (WIP)
