# Deep Neural Networks for Speech Recognition

# Getting Started 

Dependencies can be installed by invoking:

    ./bootstrap.sh
	
For training on GPUs use:

    ./bootstrap.sh --gpus
	
	
Installing dataset:

    cd datasets
	./get_datasets.sh

Training can be invoked by using:

    python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --weight_decay 0.00001 --lr 0.1 0.01 0.001  --gpu_no 0 --features raw --model ekut-raw-cnn3-1d
	
To export the trained model use:

    python3.6 -m speech_recognition.export --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --features raw --input_file trained_models/model.onnx


# TODO:
  Training:
    - Implement Depthwise Separable Convolutions
    - Implement Wavenet
    - Experiment with dilations
  
  Export:
    - Add Memory Allocator
    - Add Quantization
