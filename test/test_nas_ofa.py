import logging
import os
import platform
import subprocess
from pathlib import Path

import pytest

topdir = Path(__file__).parent.absolute() / ".."


@pytest.mark.parametrize(
    "model,epochs,random_evaluate,random_evaluate_number",
    [
        ("ofa_quant", "1", "True", "5"),
        ("ofa", "1", "True", "5"),
        ("ofa_quant", "1", "False", "5"),
        ("ofa", "1", "False", "5"),
    ],
)
def test_ofa(model, epochs, random_evaluate, random_evaluate_number):
    epochs = 1
    command_line = f"python -m hannah.train --config-name nas_ofa trainer.limit_train_batches=5 trainer.limit_val_batches=5 trainer.limit_test_batches=5 experiment_id=test_ofa nas.epochs_warmup={epochs} nas.epochs_kernel_step={epochs} nas.epochs_depth_step={epochs} nas.epochs_dilation_step={epochs} nas.epochs_width_step={epochs} nas.random_evaluate=False model={model} nas.random_evaluate={random_evaluate} nas.random_eval_number={random_evaluate_number}"

    logging.info("runing commandline %s", command_line)
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
