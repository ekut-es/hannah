from pathlib import Path

import pytest
import subprocess

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
    ],
)
def test_models(model, features):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True model={model} features={features}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.parametrize(
    "model, features, compress",
    [("tc-res8", "mfcc", "fixpoint_8_8"), ("gds", "sinc", "fixpoint_8_8")],
)
def test_distiller(model, features, compress):
    command_line = f"python -m speech_recognition.train trainer.overfit_batches=0.01 trainer.max_epochs=10 model={model} features={features} compress={compress} normalizer=fixedpoint fold_bn=9"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.parametrize(
    "model,backend", [("tc-res8", "torchmobile"), ("gds", "torchmobile")]
)
def test_backend(model, backend):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True experiment_id=test_backend backend={backend} model={model}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
