#!/bin/bash

wget  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p speech_commands_v0.02
tar xvzf -C speech_commands_v0.02 speech_commands_v0.02.tar.gz
rm -f speech_commands_v0.02.tar.gz
