import os
import sys
import shutil
import random
from ..utils import list_all_files, extract_from_download_cache
from torchvision.datasets.utils import extract_archive


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
        if train != None:
            output_train.extend(train)
            output_dev.extend(dev)
            output_test.extend(test)

        train, dev, test = NoiseDataset.getFSDnoisy_divided(config)
        if train != None:
            output_train.extend(train)
            output_dev.extend(dev)
            output_test.extend(test)
        train, dev, test = NoiseDataset.getFSD50K_divided(config)
        if train != None:
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

            FSDParts = ["audio_test", "audio_train"]

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

            FSDParts = ["audio_test", "audio_train"]
            test = list()
            dev = list()
            train = list()

            test = NoiseDataset.read_dataset_specific(
                os.path.join(kaggle_folder, "FSDFSDnoisy.audio_test")
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
                downloadfolder_tmp = os.path.join(
                    sys.argv[0].replace("speech_recognition/train.py", ""),
                    "datasets/downloads",
                )

            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)

            if not os.path.isdir(downloadfolder_tmp):
                os.makedirs(downloadfolder_tmp)
                cached_files = list()
            else:
                cached_files = list_all_files(downloadfolder_tmp, ".zip")

            if not os.path.isdir(noise_folder):
                os.makedirs(noise_folder)

            if "TUT" in noisedatasets:
                tut_target = os.path.join(
                    noise_folder, "TUT-acoustic-scenes-2017-development"
                )

                if not os.path.exists(tut_target):

                    target_cache = os.path.join(downloadfolder_tmp, "TUT")

                    for i in range(1, 10):
                        filename = (
                            "TUT-acoustic-scenes-2017-development.audio."
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
