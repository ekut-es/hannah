#!/bin/bash

NUM_WORKERS=6
EXPERIMENT_ID=noise_behavior1
GPU=0

for snr in 1000 100 50 20 10 5 0 ; do
	python -m speech_recognition.train  --experiment-id $EXPERIMENT_ID --num-workers $NUM_WORKERS --gpu-no $GPU --model tc-res8 --test-snr $snr &
        python -m speech_recognition.train  --experiment-id $EXPERIMENT_ID --num-workers $NUM_WORKERS --gpu-no $GPU --model lstm-1 --test-snr $snr &
	python -m speech_recognition.train  --experiment-id $EXPERIMENT_ID --num-workers $NUM_WORKERS --gpu-no $GPU --model tc-res8 --test-snr $snr --normalize-bits 8 --fold-bn 400 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml &
done
wait

