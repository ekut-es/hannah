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
    ],
)
def test_models(model, features):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True model={model} features={features}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.parametrize("model, features, compress", [("tc-res8", "mfcc", "fp_8_8_8")])
def test_distiller(model, features, compress):
    command_line = f"python -m speech_recognition.train trainer.overfit_batches=0.01 trainer.max_epochs=10 model={model} features={features} compress={compress} normalizer=fixedpoint"
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
        ("tc-res8", "vad", "vad_balanced"),
        ("tc-res8", "kws", ""),
    ],
)
def test_datasets(model, dataset, split):
    download_folder = os.getenv(
        "TEST_DOWNLOAD_FOLDER", "/net/rausch1/export/lucille/datasets/"
    )
    if not os.path.exists(download_folder):
        logging.warning("Could not find download folder, skipping datased tests")
        return

    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True model={model} dataset={dataset} dataset.download_folder={download_folder} dataset.data_split={split}"

    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
