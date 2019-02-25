#!/bin/bash

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn1 --gpu_no 0 --seed 1234 --lr 0.001

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn2 --gpu_no 0 --seed 1234 --lr 0.001

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn3 --gpu_no 0 --seed 1234 --lr 0.001

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn1-relu --gpu_no 0 --seed 1234 --lr 0.01

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn2-relu --gpu_no 0 --seed 1234 --lr 0.01

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn3-relu --gpu_no 0 --seed 1234 --lr 0.01

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn4-relu --gpu_no 0 --seed 1234 --lr 0.01

 python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn5-relu --gpu_no 0 --seed 1234 --lr 0.01



python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-ds-cnn-small --gpu_no 0 --window_ms 40 --stride_ms 20 --n_dct 10 --lr 0.01 --seed 1234

python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-ds-cnn-medium --gpu_no 0 --window_ms 40 --stride_ms 20 --n_dct 10 --lr 0.01 --seed 1234

python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-ds-cnn-large --gpu_no 0 --window_ms 40 --stride_ms 20 --n_dct 10  --lr 0.01 --seed 1234




python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-dnn-small --gpu_no 0 --window_ms 40 --stride_ms 40 --n_dct 10 --lr 0.001 --seed 1234

python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-dnn-medium --gpu_no 0 --window_ms 40 --stride_ms 40 --n_dct 10 --lr 0.001 --seed 1234

python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-dnn-large --gpu_no 0 --window_ms 40 --stride_ms 40 --n_dct 10  --lr 0.001 --seed 1234


python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01  --seed 1234 --model honk-cnn-trad-pool2
 
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-stride1 
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-fpool3 
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-fstride4
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-fstride8
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tpool2
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tpool3
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tstride2
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tstride4
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tstride8
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-res15
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-res26
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-res8
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-res15-narrow
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-res8-narrow
python3.6 -m speech_recognition.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-res26-narrow
