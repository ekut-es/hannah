import logging
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
from tvm import relay
import re
import os
import yaml
from dataclasses import asdict


import search_space.space as space
from hannah_tvm import config
from hannah_tvm import measure
from hannah_tvm import load


def _prepare_relay(net, input_shape):
    inp = relay.var("input", shape=input_shape)
    quant_seq = net.quantization_sequence(inp)
    quant_params = space.generate_quant_params(quant_seq)
    kwargs = {"quant_params": quant_params}
    relay_mod = net.to_relay(inp, **kwargs)

    return relay_mod


def tune(
    net, input_shape, board, measure_context, log_file, config_log=None, task_log=None
):
    try:
        relay_mod = _prepare_relay(net, input_shape)
    except Exception as e:
        print("Relay conversion failed (Tuning)")
        print(str(e))
        return
    _, params = space.generate_random_params(relay_mod)

    target = tvm.target.Target(board.target)
    target_host = tvm.target.Target(board.target_host)

    if str(target.kind) == "cuda":
        if "arch" in target.attrs:
            logging.info("Setting cuda target arch %s", target.attrs["arch"])
            autotvm.measure.measure_methods.set_cuda_target_arch(target.attrs["arch"])
        else:
            logger.warning("CUDA target has no architecture attribute")

    logging.info("Extracting tasks ...")
    # hardware_params = board.get("hardware_params", None)
    hardware_params = board.hardware_params
    if hardware_params:
        hardware_params = auto_scheduler.HardwareParams(**asdict(hardware_params))

    tasks, task_weights = auto_scheduler.extract_tasks(
        relay_mod["main"], params, target, target_host, hardware_params=hardware_params
    )

    prev_tuned_tasks = {}
    if task_log and os.path.exists(task_log):
        with open(task_log, "r") as yamlfile:
            logged_tasks = yaml.safe_load(yamlfile)

        untuned_tasks = []
        untuned_tasks_weights = []

        for task, w in zip(tasks, task_weights):
            key = re.search('"(.*)"', task.workload_key).group(0).replace('"', "")
            if key not in logged_tasks["tasks"]:
                untuned_tasks.append(task)
                untuned_tasks_weights.append(w)
            else:
                print("Task {} already tuned.".format(task.desc))
                key = re.search('"(.*)"', task.workload_key).group(0).replace('"', "")
                workload = "[" + re.search('"(?:.*)", (.*)', task.workload_key).group(1)
                speed = logged_tasks["tasks"][key]["speed"]
                prev_tuned_tasks[key] = {
                    "desc": str(task.desc),
                    "workload": str(workload),
                    "speed": str(speed),
                }

        tasks = untuned_tasks
        task_weights = untuned_tasks_weights

    if tasks:
        tuner = auto_scheduler.TaskScheduler(
            tasks,
            task_weights,
            callbacks=[space.TuningResultsCallback(log_file=[config_log, task_log])],
        )

        if task_log:
            with open(config_log, "w") as outfile:
                yaml.safe_dump(
                    {"graph": net.to_dict(), "tasks": prev_tuned_tasks},
                    outfile,
                    default_flow_style=False,
                )

        runner = measure_context.runner

        logging.info("Begin tuning...")
        # log_file = f"{board.name}.log"
        # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=len(tasks) * 600,  # TODO: CHANGE TO 600
            builder="local",
            runner=runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option)
