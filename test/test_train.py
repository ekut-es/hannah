import platform
import subprocess
import os
import logging

from pathlib import Path

import pytest

topdir = Path(__file__).parent.absolute() / ".."

# def test_tc_res8_vad():
#     command_line = "python -m speech_recognition.train --model tc-res8 --dataset vad --data_folder datasets/vad_data_balanced --n-labels 2 --batch_size=2"
#     subprocess.run(
#         command_line,
#         check=True,
#         shell=True,
#         stdout=subprocess.PIPE,
#     )


@pytest.mark.parametrize(
    "model,features",
    [
        ("tc-res8", "sinc"),
        ("gds", "sinc"),
        ("lstm", "sinc"),
        ("tc-res8", "mfcc"),
        ("gds", "mfcc"),
        ("lstm", "mfcc"),
        ("tc-res8", "spectrogram"),
        ("gds", "spectrogram"),
        ("lstm", "spectrogram"),
        ("tc-res8", "melspec"),
        ("gds", "melspec"),
        ("lstm", "melspec"),
        ("wavenet", "mfcc"),
        ("nas-tc", "mfcc"),
        ("conv-net-factory", "mfcc"),
        ("conv-net-factory", "spectrogram"),
        ("conv-net-fbgemm", "mfcc"),
        ("conv-net-trax", "mfcc"),
    ],
)
def test_models(model, features):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True model={model} features={features}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.skipif(
    platform.processor() == "ppc64le",
    reason="currently needs cpu based fft wich is not available on ppc",
)
@pytest.mark.parametrize(
    "model,backend", [("tc-res8", "torchmobile"), ("gds", "torchmobile")]
)
def test_backend(model, backend):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True experiment_id=test_backend backend={backend} model={model}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.parametrize(
    "model,dataset,split",
    [
        ("tc-res8", "snips", ""),
        #        ("tc-res8", "vad", "vad_balanced"),
        ("tc-res8", "kws", ""),
        ("tc-res8", "atrial_fibrillation", ""),
        ("tc-res8", "pamap2", ""),
    ],
)
def test_datasets(model, dataset, split):
    data_folder = os.getenv(
        "HANNAH_DATA_FOLDER", "/net/rausch1/export/lucille/datasets/"
    )
    if not os.path.exists(data_folder):
        logging.warning("Could not find data folder, skipping datased tests")
        return

    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True model={model} dataset={dataset} dataset.data_folder={data_folder} dataset.data_split={split}"
    if dataset == "pamap2":
        command_line += " features=raw"

    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


def test_2d():
    command_line = "hannah-train dataset=cifar10 features=identity trainer.gpus=[1] model=conv-net-2d trainer.max_epochs=30 scheduler.max_lr=2.5"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
