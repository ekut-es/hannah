#!/bin/bash -e 

enable_gpu=0
user_arg="--user"
create_pyenv=0
using_gpu_cluster=0
python_cmd=python3.6

while [[ $# -gt 0 ]]
do
    key=$1
    
    case $key in
	--gpu) # Install machine learning frameworks for gpu
	    enable_gpu=1
	    shift
	    ;;
	--venv) # Install into virtual environment
	    create_pyenv=1
	    user_arg=""
	    shift
	    ;;
	--global) # Install dependencies in global python path
	    create_pyenv=0
	    user_arg=""
	    shift
	    ;;
	--gpu_cluster) # Install for Lensch's gpu cluster
            create_pyenv=1
            enable_gpu=1
            using_gpu_cluster=1
            user_arg=""
            shift
	    ;;
	*)    # unknown option
	    echo "Found unknown option: $key"
	    exit 1
	    ;;
    esac
done

git submodule update --init --recursive

if [ -f /etc/redhat-release ]; then
    sudo yum install python36 python36-devel -y 
    sudo yum install freeglut-devel -y 
    sudo yum install portaudio-devel -y
else
    echo "Automatic installation of system packages not supported on this OS"
    echo "Please make sure that equivalent versions of the following packages are installed:"
    echo "python36-devel freeglut-devel portaudio-devel"
fi


if [ $create_pyenv == 1 ]; then
    if [ ! -d venv ]; then
	    echo "Installing Virtual environment"

        if [ $using_gpu_cluster == 0 ]; then
	        virtualenv venv -p python3.6
        else
            /usr/bin/python3.6 -m pip install virtualenv
            /usr/bin/python3.6 -m virtualenv venv -p python3.6
        fi
    fi
    source venv/bin/activate
else
    $python_cmd -m ensurepip $user_arg
fi

if [ $using_gpu_cluster == 1 ] || [$enable_gpu == 1]; then
	$python_cmd -m pip install -r requirements.txt
    if [ -f /etc/redhat-release ]; then
        $python_cmd -m pip install $user_arg PyAudio==0.2.11
    fi
else
    #install torch
    $python_cmd -m pip install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-linux_x86_64.whl $user_arg
    $python_cmd -m pip install torchvision==0.2.2.post3 $user_arg

    #install tensorflow
    $python_cmd -m pip install tensorflow==1.12.0 $user_arg
    
    $python_cmd -m pip install $user_arg chainmap
    $python_cmd -m pip install $user_arg cherrypy
    $python_cmd -m pip install $user_arg librosa==0.6.3
    $python_cmd -m pip install $user_arg Flask
    $python_cmd -m pip install $user_arg numpy
    $python_cmd -m pip install $user_arg Pillow
    if [ -f /etc/redhat-release ]; then
        $python_cmd -m pip install $user_arg PyAudio
    fi
    $python_cmd -m pip install $user_arg python-Levenshtein
    $python_cmd -m pip install $user_arg PyOpenGL
    $python_cmd -m pip install $user_arg PyOpenGL_accelerate
    $python_cmd -m pip install $user_arg pyttsx3
    $python_cmd -m pip install $user_arg requests
    $python_cmd -m pip install $user_arg SpeechRecognition
    $python_cmd -m pip install $user_arg git+https://github.com/daemon/pytorch-pcen
    $python_cmd -m pip install $user_arg tensorboardX
    $python_cmd -m pip install $user_arg onnx
    $python_cmd -m pip install $user_arg pyyaml
    $python_cmd -m pip install $user_arg scipy
    $python_cmd -m pip install $user_arg torchnet
    $python_cmd -m pip install $user_arg pydot
    $python_cmd -m pip install $user_arg tabulate
    $python_cmd -m pip install $user_arg pandas
    $python_cmd -m pip install $user_arg jupyter
    $python_cmd -m pip install $user_arg matplotlib
    $python_cmd -m pip install $user_arg ipywidgets
    $python_cmd -m pip install $user_arg bqplot
    $python_cmd -m pip install $user_arg pytest
    $python_cmd -m pip install $user_arg xlsxwriter
    $python_cmd -m pip install $user_arg xlrd
    $python_cmd -m pip install $user_arg gitpython
    $python_cmd -m pip install $user_arg spur
    $python_cmd -m pip install $user_arg plotly
    
    echo "\nInstallation finished!\n"
    
    if [ $create_pyenv == 1 ]; then
        echo "Dependencies have been installed in a python virtual enviornment"
        echo "to activate source the appropriate script in venv/bin"
    fi
fi
