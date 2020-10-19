import pytest
import subprocess


@pytest.mark.parametrize(
    "experiment_id, model, seed, epochs, limits_datasets, distiller, fold_bn, normalize_bits, profile",
    [
        (
            "--experiment-id ci",
            "--model tc-res8",
            "--seed 1234",
            "--fast-dev-run",
            "",
            "",
            "",
            "",
            "",
        ),
        (
            "--experiment-id ci_prof",
            "--model tc-res8",
            "--seed 1234",
            "--fast-dev-run",
            "",
            "",
            "",
            "",
            "--profile",
        ),
        (
            "",
            "--model tc-res8",
            "",
            "--n-epoch=3",
            "--limits-datasets 0.01 0.01 0.01",
            "--compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_4_4.yaml",
            "--fold-bn 1",
            "--normalize-bits 8",
            "",
        ),
        (
            "",
            "--model branchy-tc-res8",
            "",
            "--n-epoch=3",
            "--limits-datasets 0.01 0.01 0.01",
            "--compress distillation/quant_aware_train_fp/quant_aware_train_fixpoint_quant_8_6.yaml",
            "--fold-bn 1",
            "--normalize-bits 8",
            "",
        ),
    ],
)
def test_tc_res8(
    experiment_id,
    model,
    seed,
    epochs,
    limits_datasets,
    distiller,
    fold_bn,
    normalize_bits,
    profile,
):
    command_line = f"python -m speech_recognition.train {experiment_id} {model} {seed} {epochs} {limits_datasets} {distiller} {fold_bn} {normalize_bits} {profile}"
    subprocess.run(command_line, check=True, shell=True, stdout=subprocess.PIPE)


# def test_tc_res8_vad():
#     command_line = "python -m speech_recognition.train --model tc-res8 --dataset vad --data_folder datasets/vad_data_balanced --n-labels 2 --batch_size=2"
#     subprocess.run(
#         command_line,
#         check=True,
#         shell=True,
#         stdout=subprocess.PIPE,
#     )

# python -m speech_recognition.train  --gpu-no 0  --n-epochs 3 --model sinc1 --experiment-id test


@pytest.mark.parametrize(
    "experiment_id, model, seed, epochs, limits_datasets, distiller, fold_bn, normalize_bits, profile",
    [
        (
            "--experiment-id ci_sinc1",
            "--model sinc1",
            "",
            "--fast-dev-run",
            "",
            "",
            "",
            "",
            "",
        )
    ],
)
def test_sinc(
    experiment_id,
    model,
    seed,
    epochs,
    limits_datasets,
    distiller,
    fold_bn,
    normalize_bits,
    profile,
):
    command_line = f"python -m speech_recognition.train {experiment_id} {model} {seed} {epochs} {limits_datasets} {distiller} {fold_bn} {normalize_bits} {profile}"
    subprocess.run(command_line, check=True, shell=True, stdout=subprocess.PIPE)


@pytest.mark.parametrize(
    "model,backend", [("tc-res8", "torchmobile"), ("sinc1", "torchmobile")]
)
def test_backend(model, backend):
    command_line = f"python -m speech_recognition.train --model {model} --backend {backend} --fast-dev-run"
    subprocess.run(command_line, check=True, shell=True, stdout=subprocess.PIPE)
