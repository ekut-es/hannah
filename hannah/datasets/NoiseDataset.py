#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import os
import random
import shutil
import sys

from torchvision.datasets.utils import extract_archive

from ..utils import extract_from_download_cache, list_all_files


class NoiseDataset:
    def __init__(self):
        pass

    @classmethod
    def getTUT_NoiseFiles(cls, config):
        data_folder = config["data_folder"]
        noise_folder = os.path.join(data_folder, "noise_files")
        tut_folder = os.path.join(noise_folder, "TUT-acoustic-scenes-2017-development")
        if os.path.isdir(tut_folder):
            files = NoiseDataset.read_dataset_specific(tut_folder)
            return files
        return []

    @classmethod
    def getOthers_divided(cls, config):
        output_train = list()
        output_dev = list()
        output_test = list()

        train, dev, test = NoiseDataset.getFSDKaggle_divided(config)
        if train is not None:
            output_train.extend(train)
            output_dev.extend(dev)
            output_test.extend(test)

        train, dev, test = NoiseDataset.getFSDnoisy_divided(config)
        if train is not None:
            output_train.extend(train)
            output_dev.extend(dev)
            output_test.extend(test)
        train, dev, test = NoiseDataset.getFSD50K_divided(config)
        if train is not None:
            output_train.extend(train)
            output_dev.extend(dev)
            output_test.extend(test)

        return (output_train, output_dev, output_test)

    @classmethod
    def getFSDKaggle_divided(cls, config):
        data_folder = config["data_folder"]
        noise_folder = os.path.join(data_folder, "noise_files")
        kaggle_folder = os.path.join(noise_folder, "FSDKaggle")
        if os.path.isdir(kaggle_folder):

            test = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSDKaggle2018.audio_test")
            )
            train = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSDKaggle2018.audio_train")
            )
            random.shuffle(train)
            dev = train[0 : len(test)]
            train = train[len(test) :]

            return (train, dev, test)
        return (None, None, None)

    @classmethod
    def getFSD50K_divided(cls, config):
        data_folder = config["data_folder"]
        noise_folder = os.path.join(data_folder, "noise_files")
        kaggle_folder = os.path.join(noise_folder, "FSD50K")
        if os.path.isdir(kaggle_folder):

            train = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSD50K.dev_audio")
            )
            test = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSD50K.eval_audio")
            )

            random.shuffle(train)
            dev = train[0 : len(test)]
            train = train[len(test) :]

            return (train, dev, test)
        return (None, None, None)

    @classmethod
    def getFSDnoisy_divided(cls, config):
        data_folder = config["data_folder"]
        noise_folder = os.path.join(data_folder, "noise_files")
        kaggle_folder = os.path.join(noise_folder, "FSDnoisy")
        if os.path.isdir(kaggle_folder):
            test = list()
            dev = list()
            train = list()

            test = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSDnoisy18k.audio_test")
            )
            train = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSDnoisy18k.audio_train")
            )
            random.shuffle(train)
            dev = train[0 : len(test)]
            train = train[len(test) :]

            return (train, dev, test)
        return (None, None, None)

    @classmethod
    def read_dataset_specific(cls, folder):
        files = list_all_files(
            folder, ".wav", remove_file_beginning=".", file_prefix=True
        )
        files.extend(
            list_all_files(folder, ".mp3", remove_file_beginning=".", file_prefix=True)
        )
        return files

    @classmethod
    def download_noise(self, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]
        noise_folder = os.path.join(data_folder, "noise_files")
        noisedatasets = config["noise_dataset"]

        if len(noisedatasets) > 0:
            if len(downloadfolder_tmp) == 0:
                download_folder = os.path.join(data_folder, "downloads")

            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)

            if not os.path.isdir(downloadfolder_tmp):
                os.makedirs(downloadfolder_tmp)
                cached_files = list()
            else:
                cached_files = list_all_files(downloadfolder_tmp, ".zip")

            if not os.path.isdir(noise_folder):
                os.makedirs(noise_folder)

            if "Timit" in noisedatasets:
                quttimit_target = os.path.join(noise_folder, "QUT-Timit")

                if not os.path.exists(quttimit_target):

                    target_cache = os.path.join(downloadfolder_tmp, "QUT-Timit")
                    base_url = "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/"
                    url_ends = [
                        "8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip",
                        "9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip",
                        "7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip",
                        "35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip",
                        "164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip",
                        "10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip",
                    ]
                    filenames = [
                        "qutnoise.zip",
                        "qutnoisecafe.zip",
                        "qutnoisecar.zip",
                        "qutnoisehome.zip",
                        "qutnoisereverb.zip",
                        "qutnoisestreet.zip",
                    ]

                    for u, filename in zip(url_ends, filenames):
                        url = base_url + u

                        extract_from_download_cache(
                            filename,
                            url,
                            cached_files,
                            target_cache,
                            noise_folder,
                            quttimit_target,
                            no_exist_check=True,
                            clear_download=clear_download,
                        )

            if "TUT" in noisedatasets:
                tut_target = os.path.join(
                    noise_folder, "TUT-acoustic-scenes-2017-development"
                )

                if not os.path.exists(tut_target):

                    target_cache = os.path.join(downloadfolder_tmp, "TUT")

                    for i in range(1, 10):
                        filename = (
                            "TUT-acoustic-scenes-2017-development-audio-"
                            + str(i)
                            + ".zip"
                        )
                        url = (
                            "https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio."
                            + str(i)
                            + ".zip"
                        )
                        extract_from_download_cache(
                            filename,
                            url,
                            cached_files,
                            target_cache,
                            noise_folder,
                            tut_target,
                            no_exist_check=True,
                            clear_download=clear_download,
                        )

            FSDParts = ["audio_test", "audio_train", "meta"]
            datasetname = ["FSDKaggle", "FSDnoisy"]
            filename_part = ["FSDKaggle2018.", "FSDnoisy18k."]
            FSDLinks = [
                "https://zenodo.org/record/2552860/files/FSDKaggle2018.",
                "https://zenodo.org/record/2529934/files/FSDnoisy18k.",
            ]
            target_cache = os.path.join(downloadfolder_tmp, "FSD")
            for name, url, filebegin in zip(datasetname, FSDLinks, filename_part):
                if name in noisedatasets:
                    for fileend in FSDParts:
                        filename = filebegin + fileend + ".zip"
                        target_folder = os.path.join(noise_folder, name)
                        target_test_folder = os.path.join(
                            target_folder, filebegin + fileend
                        )
                        tmp_url = url + fileend + ".zip"
                        extract_from_download_cache(
                            filename,
                            tmp_url,
                            cached_files,
                            target_cache,
                            target_folder,
                            target_test_folder,
                        )

            if "FSD50K" in noisedatasets:
                filename = "50k.tar"
                url = "https://atreus.informatik.uni-tuebingen.de/seafile/f/1fe048dfbbbf49eaa9d5/?dl=1"
                target_test_folder = os.path.join(target_cache, "FSDK50K")
                extract_from_download_cache(
                    filename,
                    url,
                    cached_files,
                    target_cache,
                    target_cache,
                    target_test_folder,
                    clear_download=clear_download,
                )

                fsd50k_folder = os.path.join(noise_folder, "FSD50K")
                if not os.path.isdir(fsd50k_folder):
                    os.makedirs(fsd50k_folder)

                files = ["FSD50K.dev.zip", "FSD50K.eval.zip"]
                tfolders = ["FSD50K.dev_audio", "FSD50K.eval_audio"]
                source = target_test_folder
                for f, tf in zip(files, tfolders):
                    target = os.path.join(noise_folder, "FSD50K")
                    target_test_folder = os.path.join(target, tf)
                    if not os.path.isdir(target_test_folder):
                        print("extract from download_cache: " + str(f))
                        extract_archive(
                            os.path.join(source, f),
                            target,
                            remove_finished=clear_download,
                        )
