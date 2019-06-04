Installation
============


Installing dependencies
-----------------------

Dependencies can either be installed to your Home-Directory or to a seperate python virtual environment.
On RedHat 7 based distros (Scientific Linux 7, CentOS 7) all required dependencies should be installed by bootstrap.sh 

On other distros we need the following packages:

- python3.6 and development headers
- portaudio and development headers
- freeglut and development headers
- libav-tools 
  
On Ubuntu 16.04 these are installable with the following commands:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get -y install software-properties-common python-software-properties
    sudo add-apt-repository -y ppa:jonathonf/python-3.6
    sudo apt-get update
    
    sudo apt-get -y install libav-tools
    sudo apt-get -y install python3.6-dev freeglut3-dev portaudio19-dev
    sudo apt-get -y install git curl wget
    curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6


Setup of virtual environment (recommended)
------------------------------------------

To install in a python virtual environment use for training on cpus:

.. code-block:: bash

    ./bootstrap.sh --venv
    
or for training on gpus:

.. code-block:: bash

    ./bootstrap.sh --venv --gpu

or for training on cluster-gpu0x:

.. code-block:: bash

    ./bootstrap.sh --gpu_cluster

And activate the venv using:

.. code-block:: bash

    source venv/bin/activate

Export LD\_LIBRARY\_PATH when training on cluster-gpu0x:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_9.0/cuda/lib64/:/graphics/opt/opt_Ubuntu18.04/cuda/cudnn/7.1.4_for_9.0/cuda/lib64


Installing the datasets
-----------------------

Google Speech Recognition Dataset can be installed using:

.. code-block:: bash

    cd datasets
    ./get_datasets.sh

Hey Snips dataset can not be automatically installed as it needs,
confirmation of their license. Please follow https://github.com/snipsco/keyword-spotting-research-datasets/blob/master/README.md to obtain a download link and
make sure you can agree to the licensing terms.
