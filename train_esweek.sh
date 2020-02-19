#!/bin/bash

EXPERIMENT_ID = esweek_quantize
NUM_WORKERS=24
GPU=3
# Train for model selection



# train with quantization for results qualification
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant.yaml --gpu-no 3 &
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml --gpu-no 3 &
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_4.yaml --gpu-no 3 &
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --gpu-no 3 &
wait

