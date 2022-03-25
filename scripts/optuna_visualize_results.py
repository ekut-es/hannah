import optuna

storage = "sqlite:///trained_models/test_optuna_resume/conv_net_trax/trial.sqlite"
study_name = "test_optuna_resume"


def main():
    study = optuna.load_study(study_name, storage)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("history.png")
    fig.write_image("history.pdf")


if __name__ == "__main__":
    main()
