#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import os
import platform
import subprocess
from pathlib import Path

import pytest

topdir = Path(__file__).parent.absolute() / ".."

# def test_tc_res8_vad():
#     command_line = "python -m hannah.tools.train --model tc-res8 --dataset vad --data_folder datasets/vad_data_balanced --n-labels 2 --batch_size=2"
#     subprocess.run(
#         command_line,
#         check=True,
#         shell=True,
#         stdout=subprocess.PIPE,
#     )


@pytest.mark.integration
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
        ("conv-net-factory", "mfcc"),
        ("conv-net-factory", "spectrogram"),
        ("conv-net-fbgemm", "mfcc"),
        ("conv-net-trax", "mfcc"),
        ("conv-net-factory", "melspec"),
    ],
)
def test_models(model, features):
    command_line = f"python -m hannah.tools.train trainer.fast_dev_run=True model={model} features={features}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,backend", [("tc-res8", "torchmobile"), ("gds", "torchmobile")]
)
def test_backend(model, backend):
    command_line = f"python -m hannah.tools.train trainer.fast_dev_run=True experiment_id=test_backend backend={backend} model={model}"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,dataset,split",
    [
        ("tc-res8", "snips", ""),
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

    command_line = f"python -m hannah.tools.train trainer.fast_dev_run=True model={model} dataset={dataset} dataset.data_folder={data_folder} dataset.data_split={split}"
    if dataset == "pamap2":
        command_line += " features=raw"

    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "model", ["timm_resnet50", "timm_efficientnet_lite1", "timm_focalnet_base_srf"]
)
def test_2d(model):
    command_line = f"hannah-train module=image_classifier dataset=fake2d features=identity trainer.devices=[0] model={model}  trainer.fast_dev_run=true scheduler.max_lr=2.5 module.batch_size=2"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "model", ["timm_resnet50", "timm_efficientnet_lite1", "timm_resnet18"]
)
def test_cifar_2d(model):
    command_line = f"hannah-train module=image_classifier dataset=cifar10 features=identity trainer.devices=[0] model={model}  trainer.fast_dev_run=true scheduler.max_lr=2.5 module.batch_size=2"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.integration
@pytest.mark.skip(reason="Faster rcnn needs to much memory for builder")
def test_kitti():
    data_folder = os.getenv(
        "HANNAH_DATA_FOLDER", "/net/rausch1/export/lucille/datasets/"
    )
    if not os.path.exists(data_folder):
        logging.warning("Could not find data folder, skipping datased tests")
        return

    command_line = f"hannah-train --config-name config_object_detection dataset.data_folder={data_folder} trainer.fast_dev_run=true"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "config",
    [
        # FIXME:  beamforming does not seamm to work "config_dd_beamforming"
        "config_dd_compass_phase",
        "config_dd_direct_angle",
        "config_dd_cartesian_phase",
        "config_dd_compass",
        "config_dd_sin_cos_phase",
        "config_dd_cartesian",
        "config_dd_direct_angle_phase",
        "config_dd_sin_cos",
    ],
)
def test_directional(config):
    command_line = f"hannah-train --config-name {config} trainer.fast_dev_run=true module.batch_size=2"
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)


def test_quantization():
    command_line = (
        "hannah-train compression=quant model=tc-res8 trainer.fast_dev_run=true"
    )
    subprocess.run(command_line, shell=True, check=True, cwd=topdir)
