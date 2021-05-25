#!/bin/bash


for model in tc-res4 tc-res6 tc-res8 tc-res14 tc-res16 tc-res20; do
    python -m speech_recognition.train hydra/launcher=joblib hydra/sweeper=nevergrad hydra.launcher.n_jobs=10 \
	    optimizer=sgd scheduler=1cycle scheduler.max_lr='interval(0.1, 2.0)' model.width_multiplier=1.0,1.5,2.0,2.5 trainer.max_epochs=30 model=$model  experiment_id=search_teacher trainer.benchmark=True module.num_workers=4  -m
done
