# Hyperparameter Search

The [hydra](https://hydra.cc) based configuration management allows using multiple hydra sweeper plugins for hyperparameter optimization.

The [nevergrad sweeper](https://hydra.cc/docs/plugins/nevergrad_sweeper) is installed by default. Sweeper plugins support specifying the
hyperparameter search space on the commandline. For example the following command optimizes the learning rate and weight decay parameters
for the default optimizer (`adamw`):

     hannah-train optimizer.lr='interval(0.00001, 0.001)' optimizer.weight_decay='interval(0.00001, 0.001)' hydra/sweeper=nevergrad hydra/launcher=joblib -m

Parametrization can be given on the commandline as well as using configuration files for a detailed documentation have a look at  [basic override syntax](https://hydra.cc/docs/advanced/override_grammar/basic)  and [extended override syntax](https://hydra.cc/docs/advanced/override_grammar/extended).
