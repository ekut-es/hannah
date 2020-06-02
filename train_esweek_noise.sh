#!/bin/bash

NUM_WORKERS=6
EXPERIMENT_ID=noise_behavior1
GPU=2
SNR=inf

python -m speech_recognition.train  --experiment-id $EXPERIMENT_ID --num-workers $NUM_WORKERS --gpu-no $GPU --model tc-res8 --test-snr $SNR &

python -m speech_recognition.train  --experiment-id $EXPERIMENT_ID --num-workers $NUM_WORKERS --gpu-no $GPU --model lstm-1 --test-snr $SNR &

#python -m speech_recognition.train  --experiment-id $EXPERIMENT_ID --num-workers $NUM_WORKERS --gpu-no $GPU --model tc-res8 --test-snr $snr --normalize-bits 8 --fold-bn 400 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml &


