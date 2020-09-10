#!/bin/bash

experiment_id=default2

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn1-relu --gpu_no 0 --seed 1234 --lr 0.01 --tcml_skip_data --tcml_skip_simage  --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

 python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn2-relu --gpu_no 0 --seed 1234 --lr 0.01 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

 python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn3-relu --gpu_no 0 --seed 1234 --lr 0.01 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

 python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn4-relu --gpu_no 0 --seed 1234 --lr 0.01 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

 python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model ekut-raw-cnn5-relu --gpu_no 0 --seed 1234 --lr 0.01 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-ds-cnn-small --gpu_no 0 --window_ms 40 --stride_ms 20 --n_dct 10 --lr 0.01 --seed 1234 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --dropout_prob 0.1 --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-ds-cnn-medium --gpu_no 0 --window_ms 40 --stride_ms 20 --n_dct 10 --lr 0.01 --seed 1234 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --dropout_prob 0.1 --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-ds-cnn-large --gpu_no 0 --window_ms 40 --stride_ms 20 --n_dct 10  --lr 0.01 --seed 1234 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --dropout_prob 0.1 --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-dnn-small --gpu_no 0 --window_ms 40 --stride_ms 40 --n_dct 10 --lr 0.01 --seed 1234 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-dnn-medium --gpu_no 0 --window_ms 40 --stride_ms 40 --n_dct 10 --lr 0.01 --seed 1234 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500 --model hello-dnn-large --gpu_no 0 --window_ms 40 --stride_ms 40 --n_dct 10  --lr 0.01 --seed 1234 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id


python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01  --seed 1234 --model honk-cnn-trad-pool2 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-stride1 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-fpool3  --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-fstride4 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-one-fstride8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tpool2 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tpool3 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tstride2 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tstride4 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.01 --seed 1234 --model honk-cnn-tstride8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model honk-res15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model honk-res26 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model honk-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model honk-res15-narrow --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model honk-res8-narrow --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00  --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model honk-res26-narrow --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id

python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --experiment_id=$experiment_id
