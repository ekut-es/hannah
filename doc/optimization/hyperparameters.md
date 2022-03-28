# Hyperparameter Search

The [hydra](https://hydra.cc) based configuration management allows using multiple hydra sweeper plugins for hyperparameter optimization.

The [optuna sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper/) is installed by default. Sweeper plugins support specifying the
hyperparameter search space on the commandline. For example the following command optimizes the learning rate and weight decay parameters
for the default optimizer (`adamw`):

     hannah-train scheduler.max_lr='interval(0.00001, 0.001)' optimizer.weight_decay='interval(0.00001, 0.001)' hydra/sweeper=optuna hydra/launcher=joblib -m

Parametrization can be given on the commandline as well as using configuration files for a detailed documentation have a look at  [basic override syntax](https://hydra.cc/docs/advanced/override_grammar/basic)  and [extended override syntax](https://hydra.cc/docs/advanced/override_grammar/extended).


## Optuna Options

The optuna hyperparameter optimizer has the following options.

```bash
hannah-train hydra/sweeper=optuna --cfg hydra -p hydra.sweeper
# @package hydra.sweeper
sampler:
  _target_: optuna.samplers.TPESampler
  seed: null
  consider_prior: true
  prior_weight: 1.0
  consider_magic_clip: true
  consider_endpoints: false
  n_startup_trials: 10
  n_ei_candidates: 24
  multivariate: false
  warn_independent_sampling: true
_target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
direction: minimize
storage: null
study_name: null
n_trials: 20
n_jobs: 2
search_space: {}
```

Try especially increasing the number of trials to run `n_trials`, and if the sweeper options are dependent on each other try enabling multivariate sampling e.g. `sampler.multivariate`.

## State Persistence

If you want to save the optimization results for later analyis you can save them to a relational or redis database by setting
the `study_name` and `storage` options. e.g.:

```
hannah-train experiment_id="test_optuna_resume" hydra/sweeper=optuna hydra.sweeper.study_name='${experiment_id}' hydra.sweeper.storage='sqlite:///${experiment_id}.sqlite'  trainer.max_epochs=15 scheduler.max_lr='interval(0.0001, 0.1)' module.num_workers=4 -m
```

The tuning results can then be visualized using the tools from [`optuna.visualize`](https://optuna.readthedocs.io/en/stable/search.html?q=visualize&check_keywords=yes&area=default#):
A simple example would be:

```python
import optuna

storage = "sqlite:///test_optuna_resume.sqlite"
study_name = "test_optuna_resume"

def main():
    study = optuna.load_study(study_name, storage)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("history.png")
    fig.write_image("history.pdf")


if __name__ == '__main__':
    main()
```

For a slightly extended visualization script have a look at `scripts/optuna_visualize_results.py` or use [`optuna-dashboard`](https://github.com/optuna/optuna-dashboard) for a web based visualization.

## Resuming Trials
An additional benefit of exporting the study state to database is that trials can be somewhat resumed at a later time by just restarting the trial with the same parameters.
Although this has a view problems:
- The hydra sweeper will start it's job numbering from scratch


## Running on ml-cloud

It is possible to use the sqlite backend to store trials on ml-cloud for an example look at  `scripts/ml_cloud_hyperopt.sh` but sqlite might be somewhat unstabble when running multiple workers.
