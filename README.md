# Deep Neural Networks for Speech Recognition

# Getting Started 


## Installing dependencies

Dependencies can either be installed to your Home-Directory or to a seperate python virtual environment.

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
   
    python3.6 -m speech_recognition.train  --no_cuda  --model ekut-raw-cnn3-1d-relu

Training on 1st GPU can be invoked by:

    python3.6 -m speech_recognition.train  --gpu_no 0  --model ekut-raw-cnn3-1d-relu


# Exporting Models for RISC-V
	
The export is currently not available. 


# TODO:
Training:
  
- Implement Wavenet
- Experiment with dilations
  
Export:
- Use relay IR
- 2D Convolutions
- Average Pooling
- Dilations
- Depthwise separable convolutions
- Batch normalization
- Add Memory Allocator
- Add Quantization support
