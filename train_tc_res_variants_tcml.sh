#!/bin/bash

id=mfcc
feature=mfcc
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id
 
id=spectrogram
feature=spectrogram
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
 
id=melspec
feature=melspec
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --tcml-mem=6G
 
 
id=fully_convolutional
feature=mel
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --fully-convolutional
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --fully-convolutional
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --fully-convolutional
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --fully-convolutional
 
 
id=20_mfccs
feature=mel
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 20 --n_mfcc 20
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 20 --n_mfcc 20  
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 20 --n_mfcc 20 
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 20 --n_mfcc 20 
 
 
id=30_mfccs
feature=mel
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 30 --n_mfcc 30
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 30 --n_mfcc 30  
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 30 --n_mfcc 30 
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id	  --n_mels 30 --n_mfcc 30 
 
 
id=500msec
feature=mel
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --input_length 8000
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --input_length 8000
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --input_length 8000
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --input_length 8000
 
 
id=8000Hz
feature=mel
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --samplingrate 8000
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res8-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --samplingrate 8000
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --samplingrate 8000
 
python -m speech_recognition.tcml.train --data_folder datasets/speech_commands_v0.02/ --wanted_words yes no up down left right on off stop go --n_labels 12 --n_epochs 500  --gpu_no 0  --lr 0.1 --seed 1234 --model tc-res14-15 --tcml_skip_data --tcml_skip_simage --tcml_skip_code --tcml_estimated_time 12:00:00 --tcml_partition day --features $feature --experiment-id $id --samplingrate 8000
