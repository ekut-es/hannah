#!/bin/bash

for bits in 2 4 6; do
    python -m hannah.train trainer.max_epochs=50  model=conv-net-trax  trainer.gpus=[1]  module.num_workers=4 normalizer=fixedpoint model.qconfig.config.noise_prob=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 experiment_id=sweep_quant_noise_pot_${bits}bit model.qconfig.config.bw_w=$bits model.qconfig.config.power_of_2=true hydra/launcher=joblib hydra.launcher.n_jobs=5 -m
done
