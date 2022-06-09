# Installation Notes

## Mac OS

**Caution**: MacOS is not tested on CI, and might break at any point in time.

Possible Ways to install all the needed Packages for Mac are with Homebrew
Python 3.9 should be installed

    brew install python@3.9

#### Mac with x86_64 processor

poetry environment should be created with

    poetry install

#### Mac with m1 processor

Install hdf5 should be installed

    brew install hdf5

Install the h5py python Package manually

    poetry run pip3 install h5py==3.6.0

poetry environment should be created with

    poetry install

Environment installed successfully
