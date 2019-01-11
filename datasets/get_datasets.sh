#!/bin/bash

mkdir -p speech_commands_v0.02

pushd speech_commands_v0.02
wget  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar xvzf  speech_commands_v0.02.tar.gz
rm -f speech_commands_v0.02.tar.gz
popd 
