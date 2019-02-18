#!/bin/bash

enable_gpu=0

while [[ $# -gt 0 ]]
do
    key=$1
    
    case $key in
	--gpu) # Install machine learning frameworks for gpu
	    enable_gpu=1
	    shift
	    ;;
	*)    # unknown option
	    echo "Found unknown option: $key"
	    exit 1
	    ;;
    esac
done

git submodule update --init --recursive

sudo yum install python36 python36-devel -y 
sudo yum install freeglut-devel -y 
sudo yum install portaudio-devel -y

python3.6 -m ensurepip --user

#install torch
if [ $enable_gpu == 0 ]; then
    python3.6 -m pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl --user
else
    python3.6 -m pip install torch --user
fi

python3.6 -m pip install torchvision --user


#install tensorflow
if [ $enable_gpu == 0 ]; then
    python3.6 -m pip install tensorflow --user
else
    python3.6 -m pip install tensorflow-gpu --user
fi

python3.6 -m pip install --user chainmap
python3.6 -m pip install --user cherrypy
python3.6 -m pip install --user librosa
python3.6 -m pip install --user Flask
python3.6 -m pip install --user numpy
python3.6 -m pip install --user Pillow
python3.6 -m pip install --user PyAudio
python3.6 -m pip install --user PyOpenGL
python3.6 -m pip install --user PyOpenGL_accelerate
python3.6 -m pip install --user pyttsx3
python3.6 -m pip install --user requests
python3.6 -m pip install --user SpeechRecognition
python3.6 -m pip install --user git+https://github.com/daemon/pytorch-pcen
python3.6 -m pip install --user tensorboardX
python3.6 -m pip install --user onnx
python3.6 -m pip install --user pyyaml
python3.6 -m pip install --user scipy
python3.6 -m pip install --user torchnet
python3.6 -m pip install --user pydot
python3.6 -m pip install --user tabulate
python3.6 -m pip install --user pandas
python3.6 -m pip install --user jupyter
python3.6 -m pip install --user matplotlib
python3.6 -m pip install --user ipywidgets
python3.6 -m pip install --user bqplot
python3.6 -m pip install --user pytest
python3.6 -m pip install --user xlsxwriter
python3.6 -m pip install --user gitpython
