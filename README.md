# Deep Neural Networks for Speech Recognition

# Getting Started 


## Installing dependencies

Dependencies can either be installed to your Home-Directory or to a seperate pyenv.

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
   
    python3.6 -m speech_recognition.train  --gpu_no 0  --model ekut-raw-cnn3-1d-relu

Training on 1st GPU can be invoked by:

    python3.6 -m speech_recognition.train  --gpu_no 0  --model ekut-raw-cnn3-1d-relu


# Exporting Models for RISC-V
	

	
To export the trained model use:

    python3.6 -m speech_recognition.export --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --features raw --input_file trained_models/model.onnx



# TODO:
  Training:
  
- Implement Wavenet
- Experiment with dilations
  
  Export:
  
- 2D Convolutions
- Average Pooling
- Dilations
- Depthwise separable convolutions
- Batch normalization
- Add Memory Allocator
- Add Quantization support
