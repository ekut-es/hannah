# Hyperparameter Search

The [hydra](https://hydra.cc) based configuration management allows using multiple hydra sweeper plugins for hyperparameter optimization.

The [nevergrad sweeper](https://hydra.cc/docs/plugins/nevergrad_sweeper) is installed by default. Sweeper plugins support specifying the
hyperparameter search space on the commandline. For example the following command optimizes the learning rate and weight decay parameters
for the default optimizer (`adamw`):

     python -m speech_recognition.train optimizer.lr='interval(0.00001, 0.001)' optimizer.weight_decay='interval(0.00001, 0.001)' hydra/sweeper=nevergrad hydra/launcher=joblib -m
