#!/bin/bash

sudo yum groupinstall -y 'Development Tools' && \
  sudo yum install -y epel-release && \
  sudo yum install -y golang libseccomp-devel \
  squashfs-tools cryptsetup

echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
  echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
  source ~/.bashrc

VERSION=3.7.3
wget https://github.com/hpcng/singularity/releases/download/v$VERSION/singularity-$VERSION.tar.gz
tar xvf singularity-$VERSION.tar.gz
cd singularity
./mconfig
cd ./builddir
make
sudo make install
