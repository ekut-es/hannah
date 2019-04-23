# Deep Neural Networks for Speech Recognition

# Getting Started 


## Installing dependencies

Dependencies can either be installed to your Home-Directory or to a seperate python virtual environment.
On RedHat 7 based distros (Scientific Linux 7, CentOS 7) all required dependencies should be installed by bootstrap.sh 

On other distros we need the following packages:

- python3.6 and development headers
- portaudio and development headers
- freeglut and development headers

On Ubuntu 16.04 these are installable with the following commands:

    sudo apt-get update
    sudo apt-get -y install software-properties-common python-software-properties
    sudo add-apt-repository -y ppa:jonathonf/python-3.6
    sudo apt-get update
    
    sudo apt-get -y install python3.6-dev freeglut3-dev portaudio19-dev
    sudo apt-get -y install git curl wget
    curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6


### Setup of virtual environment (recommended)

To install in a python virtual environment use for training on cpus:

    ./bootstrap.sh --venv
    
or for training on gpus:

    ./bootstrap.sh --venv --gpu

And activate the venv using:

    source venv/bin/activate
    
    
### Setup with installation to home directory

Dependencies can be installed by invoking:

    ./bootstrap.sh
	
For training on GPUs use:

    ./bootstrap.sh --gpu
    
	
## Installing the datasets
	
Installing dataset:

    cd datasets
	./get_datasets.sh

## Training

Training on CPU can be invoked by:
   
    python3.6 -m speech_recognition.train  --no_cuda  --model ekut-raw-cnn3-relu

Training on 1st GPU can be invoked by:

    python3.6 -m speech_recognition.train  --gpu_no 0  --model ekut-raw-cnn3-relu

## Evaluation




# Exporting Models for RISC-V
	
The export is currently not available. 


# TODO:
Training:
  
- Implement Wavenet
- Experiment with dilations
- Add design space exploration tool (WIP)
- Add estimation of non functional properties on algorithmic level (WIP)

Export ISA / RT-Level:
- Make usable again
- Use relay IR / TVM
- 2D Convolutions
- Average Pooling
- Dilations
- Depthwise separable convolutions
- Batch normalization
- Add Memory Allocator
- Add Quantization support
