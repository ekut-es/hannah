import logging


def extract():
    for name, source in data.items():
        print(f"Extracting design points for {name}")
        output_folder = Path("model")
        output_folder.mkdir(exist_ok=True, parents=True)
        history_path = Path(source) / "history.yml"
        with history_path.open("r") as f:
            history = yaml.unsafe_load(f)

        results = (h.result for h in history)
        metrics = pd.DataFrame(results)
        pareto_points = is_pareto(metrics.to_numpy())

        metrics["is_pareto"] = pareto_points

        candidates = metrics[metrics["is_pareto"]]
        candidates = candidates[candidates["val_error"] < bounds[name]["val_error"]]

        for point, metric in [
            ("ha", "val_error"),
            ("lp", "acc_power"),
            ("la", "acc_area"),
        ]:
            sorted = candidates.sort_values(metric)
            print(sorted)

            for num, (index, metrics) in enumerate(sorted.head(3).iterrows()):
                print(metric, num, index, metrics)
                parameters = history[int(index)].parameters.flatten()
                metrics = history[int(index)].result
                model_parameters = parameters["model"]
                backend_parameters = parameters["backend"]

                file_name = f"nas2_{name.lower()}_{point}_top{num}.yaml"
                file_path = output_folder / file_name
                with file_path.open("w") as f:
                    f.write("# @package _group_\n")
                    f.write("\n")
                    f.write("# Backend parameters:\n")
                    for k, v in backend_parameters.items():
                        f.write(f"#   backend.{k}={v}\n")
                    f.write("\n")
                    f.write("# Expected results:")
                    for k, v in metrics.items():
                        f.write(f"#   {k}: {v}\n")
                    f.write("\n")
                    f.write(
                        "_target_: speech_recognition.models.factory.factory.create_cnn\n"
                    )
                    f.write(f"name: nas2_{name.lower()}_{point}_top{num}\n")
                    f.write("norm:\n")
                    f.write("  target: bn\n")
                    f.write("act:\n")
                    f.write("  target: relu\n")

                    model_parameters["qconfig"][
                        "_target_"
                    ] = "speech_recognition.models.factory.qconfig.get_trax_qat_qconfig"
                    model_parameters["qconfig"]["config"]["power_of_2"] = False
                    model_parameters["qconfig"]["config"]["noise_prob"] = 0.7

                    f.write(yaml.dump(model_parameters))
