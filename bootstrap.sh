#!/bin/bash

sudo yum install python36 python36-devel -y 
sudo yum install freeglut-devel -y 
sudo yum install portaudio-devel -y

python3.6 -m ensurepip --user

#install torch
python3.6 -m pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl --user
#python3.6 -m pip install torch --user
python3.6 -m pip install torchvision --user


#install tensorflow
python3.6 -m pip install tensorflow --user
#python3.6 -m pip install tensorflow-gpu --user


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


