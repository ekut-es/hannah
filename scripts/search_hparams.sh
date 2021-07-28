#!/bin/bash

for model in tc-res4 tc-res8 conv-net-fbgemm conv-net-trax; do
    python -m hannah.train hydra/sweeper=nevergrad hydra/launcher=joblib optimizer.lr='interval(0.00001, 0.001)' optimizer.weight_decay='interval(0.000001, 0.001)' hydra.launcher.n_jobs=5 trainer.gpus=1 module.num_workers=2 early_stopping=default -m
done
