#!/bin/bash

wget  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar xvzf -C speech_commands speech_commands_v0.02.tar.gz
rm -f speech_commands_v0.02.tar.gz
