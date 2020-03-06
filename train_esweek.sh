#!/bin/bash

EXPERIMENT_ID=exploration
NUM_WORKERS=24
GPU=1

# Train for model selection
for MODEL in tc-res2 tc-res4 tc-res6 tc-res8 tc-res10  tc-res12  tc-res14 tc-res16  tc-res18  tc-res20; do
  python3.6 -m speech_recognition.train --experiment-id $EXPERIMENT_ID --num_workers 24 --gpu_no $GPU --model $MODEL &
done

GPU=0

# train with quantization for results qualification
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant.yaml --gpu-no $GPU &
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml --gpu-no $GPU &
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_4.yaml --gpu-no $GPU &
python3.6 -m speech_recognition.train --experiment-id fp_norm_fold --num-workers 24 --normalize-bits 8 --fold-bn 400 --model tc-res8 --gpu-no $GPU &
wait

