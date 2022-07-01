#!/bin/sh

config="nas_kws_kd nas_kws_mfcc nas_kws_spec nas_kws nas_snips" #  nas_vad"

for c in $config; do 
  sbatch --job-name $c ./scripts/ml_cloud_nas.sh $c
done

