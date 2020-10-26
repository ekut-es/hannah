from pathlib import Path

import pytest
import subprocess

topdir = Path(__file__).parent.absolute() / ".."

# @pytest.mark.parametrize(
#     "experiment_id, model, seed, epochs, limits_datasets, distiller, fold_bn, normalize_bits, profile",
#     [
#         (
#             "--experiment-id ci",
#             "--model tc-res8",
#             "--seed 1234",
#             "--fast-dev-run",
#             "",
#             "",
#             "",
#             "",
#             "",
#         ),
#         (
#             "--experiment-id ci_prof",
#             "--model tc-res8",
#             "--seed 1234",
#             "--fast-dev-run",
#             "",
#             "",
#             "",
#             "",
#             "--profile",
#         ),
#         (
#             "",
#             "--model tc-res8",
#             "",
#             "--n-epoch=3",
#             "--limits-datasets 0.01 0.01 0.01",
#             "--compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_4_4.yaml",
#             "--fold-bn 1",
#             "--normalize-bits 8",
#             "",
#         ),
#         (
#             "",
#             "--model branchy-tc-res8",
#             "",
#             "--n-epoch=3",
#             "--limits-datasets 0.01 0.01 0.01",
#             "--compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml",
#             "--fold-bn 1",
#             "--normalize-bits 8",
#             "",
#         ),
#     ],
# )
# def test_tc_res8(
#     experiment_id,
#     model,
#     seed,
#     epochs,
#     limits_datasets,
#     distiller,
#     fold_bn,
#     normalize_bits,
#     profile,
# ):
#     command_line = f"python -m speech_recognition.train {experiment_id} {model} {seed} {epochs} {limits_datasets} {distiller} {fold_bn} {normalize_bits} {profile}"
#     subprocess.run(command_line, check=True, shell=True, stdout=subprocess.PIPE)


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
    [("tc-res8", "sinc"), ("sinc1", "sinc"), ("tc-res8", "mfcc"), ("sinc1", "mfcc")],
)
def test_models(model, features):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True model={model} features={features}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.parametrize(
    "model, features, compress",
    [("tc-res8", "mfcc", "fixpoint_8_8"), ("sinc1", "sinc", "fixpoint_8_8")],
)
def test_distiller(model, features, compress):
    command_line = f"python -m speech_recognition.train trainer.overfit_batches=0.01 trainer.max_epochs=10 model={model} features={features} compress={compress} normalizer=fixedpoint fold_bn=9"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.parametrize(
    "model,backend", [("tc-res8", "torchmobile"), ("sinc1", "torchmobile")]
)
def test_backend(model, backend):
    command_line = f"python -m speech_recognition.train trainer.fast_dev_run=True experiment_id=test_backend backend={backend} model={model}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
