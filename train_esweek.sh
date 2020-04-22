#!/bin/bash

EXPERIMENT_ID=exploration
NUM_WORKERS=24
GPU=1

BASE_MODELS="tc-res2 tc-res4 tc-res6 tc-res8 tc-res10  tc-res12  tc-res14 tc-res16  tc-res18  tc-res20"

# Train for model selection
#for MODEL in ${BASE_MODELS}; do
#  python3.6 -m speech_recognition.train --experiment-id $EXPERIMENT_ID --num_workers $NUM_WORKERS --gpu_no $GPU --model $MODEL &
#done
#GPU=2
#for MODEL in ${BASE_MODELS}; do
#  python3.6 -m speech_recognition.train --width_multiplier=1.5 --experiment-id $EXPERIMENT_ID --num_workers $NUM_WORKERS --gpu_no $GPU --model $MODEL &
#done
#GPU=3
#for MODEL in ${BASE_MODELS}; do
#  python3.6 -m speech_recognition.train --width_multiplier=0.75 --experiment-id $EXPERIMENT_ID --num_workers $NUM_WORKERS --gpu_no $GPU --model $MODEL &
#done

 
GPU=0
 
# train with quantization for results qualification
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant.yaml --gpu-no $GPU &
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_16.yaml --gpu-no $GPU &
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml --gpu-no $GPU &
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_4.yaml --gpu-no $GPU &
 
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 16 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_16_16.yaml --gpu-no $GPU &
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 6 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_6_6.yaml --gpu-no $GPU &
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 4 --fold-bn 400 --model tc-res8 --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_4_4.yaml --gpu-no $GPU &
 
#python3.6 -m speech_recognition.train --dump-test --experiment-id fp_norm_fold --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 400 --model tc-res8 --gpu-no $GPU &
#wait


# Search Branchy Resnet Configuration

#for threshold in 0.6 0.8 0.9 1.0 1.2 1.4 1.6; do
#    python3.6 -m speech_recognition.train --dump-test --experiment-id branchy_search --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 400 --model branchy-tc-res8 --gpu-no $GPU  --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant.yaml --earlyexit_thresholds $threshold  --earlyexit_lossweights 0.4 &
#done


#for threshold in 0.3 0.4 0.5 0.6 0.7 0.8; do
#    python3.6 -m speech_recognition.train --dump-test --experiment-id branchy_search-2exits --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 450 --model branchy-tc-res8 --gpu-no $GPU  --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant.yaml --earlyexit_thresholds ${threshold} ${threshold}  --earlyexit_lossweights 0.3 0.3 &
#done

#for threshold in 0.3 0.4 0.5 0.6 0.7 0.8; do
#    python3.6 -m speech_recognition.train --dump-test --experiment-id branchy_search-2exits_8_6 --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 450 --model branchy-tc-res8 --gpu-no $GPU  --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml --earlyexit_thresholds ${threshold} ${threshold}  --earlyexit_lossweights 0.3 0.3 &
#done

python3.6 -m speech_recognition.train --dump-test --experiment-id branchy_search-2exits_8_6 --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 450 --model branchy-tc-res8 --gpu-no $GPU  --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml --earlyexit_thresholds 0.9 0.9  --earlyexit_lossweights 0.3 0.3 &

python3.6 -m speech_recognition.train --dump-test --experiment-id branchy_search-2exits_8_6 --num-workers $NUM_WORKERS --normalize-bits 8 --fold-bn 450 --model branchy-tc-res8 --gpu-no $GPU  --compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml --earlyexit_thresholds 1.0 1.0  --earlyexit_lossweights 0.3 0.3 &

wait


