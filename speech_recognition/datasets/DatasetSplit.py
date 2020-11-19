import os
import random
import shutil
import numpy as np
from ..utils import list_all_files, extract_from_download_cache
from .NoiseDataset import NoiseDataset
from torchvision.datasets.utils import list_dir
from pandas import DataFrame
import pandas as pd


class DatasetSplit:
    def __init__(self):
        pass

    @classmethod
    def vad(cls, config):
        # directories with original data
        data_folder = config["data_folder"]
        noise_dir = os.path.join(data_folder, "noise_files")
        speech_dir = os.path.join(data_folder, "speech_files")

        noise_files = NoiseDataset.getTUT_NoiseFiles(config)
        noise_files_others = NoiseDataset.getOthers_divided(config)

        # list all noise  and speech files
        speech_files = DatasetSplit.read_UWNU(config)

        # randomly shuffle the noise and speech files and split them in train,
        # validation and test set
        random.shuffle(noise_files)
        random.shuffle(speech_files)

        nb_noise_files = len(noise_files)
        nb_train_noise = int(0.6 * nb_noise_files)
        nb_dev_noise = int(0.2 * nb_noise_files)

        nb_speech_files = len(speech_files)
        nb_train_speech = int(0.6 * nb_speech_files)
        nb_dev_speech = int(0.2 * nb_speech_files)

        train_noise = noise_files[:nb_train_noise]
        dev_noise = noise_files[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_noise = noise_files[nb_train_noise + nb_dev_noise :]

        train_other, dev_other, test_other = noise_files_others
        train_noise.extend(train_other)
        dev_noise.extend(dev_other)
        test_noise.extend(test_other)

        nb_noise_files += len(test_other)
        nb_train_noise += len(train_other)
        nb_dev_noise += len(dev_other)

        train_speech = speech_files[:nb_train_speech]
        dev_speech = speech_files[nb_train_speech : nb_train_speech + nb_dev_speech]
        test_speech = speech_files[nb_train_speech + nb_dev_speech :]

        mozilla = DatasetSplit.read_mozilla(config)
        mozilla_train, mozilla_dev, mozilla_test = mozilla

        train_speech.extend(mozilla_train)
        dev_speech.extend(mozilla_dev)
        test_speech.extend(mozilla_test)

        destination_dict = {
            "train/noise": train_noise,
            "train/speech": train_speech,
            "dev/noise": dev_noise,
            "dev/speech": dev_speech,
            "test/noise": test_noise,
            "test/speech": test_speech,
        }

        return destination_dict

    @classmethod
    def vad_balanced(cls, config):
        # directories with original data
        data_folder = config["data_folder"]
        noise_dir = os.path.join(data_folder, "noise_files")
        speech_dir = os.path.join(data_folder, "speech_files")

        noise_files = NoiseDataset.getTUT_NoiseFiles(config)
        noise_files_others = NoiseDataset.getOthers_divided(config)

        speech_files = DatasetSplit.read_UWNU(config)

        # randomly shuffle noise and speech files and split them in train,
        # validation and test set
        random.shuffle(noise_files)
        random.shuffle(speech_files)

        nb_noise_files = len(noise_files)
        nb_train_noise = int(0.6 * nb_noise_files)
        nb_dev_noise = int(0.2 * nb_noise_files)

        train_noise = noise_files[:nb_train_noise]
        dev_noise = noise_files[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_noise = noise_files[nb_train_noise + nb_dev_noise :]

        train_other, dev_other, test_other = noise_files_others
        train_noise.extend(train_other)
        dev_noise.extend(dev_other)
        test_noise.extend(test_other)

        train_speech = speech_files[:nb_train_noise]
        dev_speech = speech_files[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_speech = speech_files[nb_train_noise + nb_dev_noise : nb_noise_files]

        mozilla = DatasetSplit.read_mozilla(config)
        mozilla_train, mozilla_dev, mozilla_test = mozilla

        random.shuffle(mozilla_train)
        random.shuffle(mozilla_dev)
        random.shuffle(mozilla_test)

        train_speech.extend(mozilla_train[0 : len(train_other)])
        dev_speech.extend(mozilla_dev[0 : len(dev_other)])
        test_speech.extend(mozilla_test[0 : len(test_other)])

        random.shuffle(train_noise)
        random.shuffle(dev_noise)
        random.shuffle(test_noise)

        train_bg_noise = train_noise[:100]
        dev_bg_noise = dev_noise[:100]
        test_bg_noise = test_noise[:100]

        destination_dict = {
            "train/noise": train_noise,
            "train/speech": train_speech,
            "dev/noise": dev_noise,
            "dev/speech": dev_speech,
            "test/noise": test_noise,
            "test/speech": test_speech,
            "train/background_noise": train_bg_noise,
            "dev/background_noise": dev_bg_noise,
            "test/background_noise": test_bg_noise,
        }

        return destination_dict

    @classmethod
    def getrennt(cls, config):
        # directories with original data
        data_folder = config["data_folder"]
        noise_dir = os.path.join(data_folder, "noise_files")
        speech_dir = os.path.join(data_folder, "speech_files")
        speech_dir = os.path.join(speech_dir, "uwnu-v2")

        # list all noise  and speech files
        noise_files = list_all_files(noise_dir, ".wav", True)
        noise_files.extend(list_all_files(noise_dir, ".mp3", True))
        speech_files_P = []
        speech_files_N = []
        for path, subdirs, files in os.walk(speech_dir):
            for name in files:
                if name.endswith("wav") and not name.startswith("."):
                    if "NC" in name:
                        speech_files_P.append(os.path.join(path, name))
                    else:
                        speech_files_N.append(os.path.join(path, name))

        # randomly shuffle noise and speech files and split them in train,
        # validation and test set

        random.shuffle(noise_files)
        random.shuffle(speech_files_P)

        nb_noise_files = len(noise_files)
        nb_train_noise = int(0.6 * nb_noise_files)
        nb_dev_noise = int(0.2 * nb_noise_files)

        train_noise = noise_files[:nb_train_noise]
        dev_noise = noise_files[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_noise = noise_files[nb_train_noise + nb_dev_noise :]

        train_speech = speech_files_N[:nb_train_noise]
        dev_speech = speech_files_P[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_speech = speech_files_P[nb_train_noise + nb_dev_noise : nb_noise_files]

        train_bg_noise = train_noise[:100]
        dev_bg_noise = dev_noise[:100]
        test_bg_noise = test_noise[:100]

        destination_dict = {
            "train/noise": train_noise,
            "train/speech": train_speech,
            "dev/noise": dev_noise,
            "dev/speech": dev_speech,
            "test/noise": test_noise,
            "test/speech": test_speech,
            "train/background_noise": train_bg_noise,
            "dev/background_noise": dev_bg_noise,
            "test/background_noise": test_bg_noise,
        }

        return destination_dict

    @classmethod
    def vad_speech(cls, config):
        # directories with original data
        data_folder = config["data_folder"]
        noise_dir = os.path.join(data_folder, "noise_files")
        speech_dir = os.path.join(data_folder, "speech_commands_v0.02")

        # list all noise  and speech files
        noise_files = NoiseDataset.getTUT_NoiseFiles(config)
        noise_files_others = NoiseDataset.getOthers_divided(config)

        speech_files = []
        for path, subdirs, files in os.walk(speech_dir):
            if "noise" not in subdirs:
                for name in files:
                    if name.endswith("wav") and not name.startswith("."):
                        speech_files.append(os.path.join(path, name))

        # randomly shuffle noise and speech files and split them in train,
        # validation and test set
        random.shuffle(noise_files)
        random.shuffle(speech_files)

        nb_noise_files = len(noise_files)
        nb_train_noise = int(0.6 * nb_noise_files)
        nb_dev_noise = int(0.2 * nb_noise_files)

        train_noise = noise_files[:nb_train_noise]
        dev_noise = noise_files[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_noise = noise_files[nb_train_noise + nb_dev_noise :]

        train_other, dev_other, test_other = noise_files_others
        train_noise.extend(train_other)
        dev_noise.extend(dev_other)
        test_noise.extend(test_other)

        nb_noise_files += len(test_other)
        nb_train_noise += len(train_other)
        nb_dev_noise += len(dev_other)

        train_speech = speech_files[:nb_train_noise]
        dev_speech = speech_files[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_speech = speech_files[nb_train_noise + nb_dev_noise : nb_noise_files]

        train_bg_noise = train_noise[:100]
        dev_bg_noise = dev_noise[:100]
        test_bg_noise = test_noise[:100]

        destination_dict = {
            "train/noise": train_noise,
            "train/speech": train_speech,
            "dev/noise": dev_noise,
            "dev/speech": dev_speech,
            "test/noise": test_noise,
            "test/speech": test_speech,
            "train/background_noise": train_bg_noise,
            "dev/background_noise": dev_bg_noise,
            "test/background_noise": test_bg_noise,
        }

        return destination_dict

    @classmethod
    def split_data(cls, config):
        data_split = config["data_split"]
        splits = ["vad", "vad_speech", "vad_balanced", "getrennt"]
        split_methods = [
            DatasetSplit.vad,
            DatasetSplit.vad_speech,
            DatasetSplit.vad_balanced,
            DatasetSplit.getrennt,
        ]

        if data_split in splits:
            print("split data begins")
            data_folder = config["data_folder"]
            target_folder = os.path.join(data_folder, data_split)

            # remove old folders
            for name in ["train", "dev", "test"]:
                oldpath = os.path.join(target_folder, name)
                if os.path.isdir(oldpath):
                    shutil.rmtree(oldpath)

            destination_dict = split_methods[splits.index(data_split)](config)

            dest_dir = os.path.join(data_folder, data_split)

            for key, value in destination_dict.items():
                data_dir = os.path.join(dest_dir, key)
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                for f in value:
                    shutil.copy2(f, data_dir)

            if config["clear_split"]:
                # remove old folders
                for name in [
                    "noise_files",
                    "speech_files",
                    "speech_commands_v0.02",
                    "hey_snips_research_6k_en_train_eval_clean_ter",
                ]:
                    oldpath = os.path.join(data_folder, name)
                    if os.path.isdir(oldpath):
                        shutil.rmtree(oldpath)

    @classmethod
    def read_UWNU(cls, config):
        data_folder = config["data_folder"]
        speech_folder = os.path.join(data_folder, "speech_files")
        uwnu_folder = os.path.join(speech_folder, "uwnu-v2")

        if os.path.isdir(uwnu_folder):

            output = list_all_files(uwnu_folder, ".wav", True, ".")
            output.extend(list_all_files(uwnu_folder, ".mp3", True, "."))

            return output
        return []

    @classmethod
    def read_mozilla(cls, config):
        data_folder = config["data_folder"]
        speech_folder = os.path.join(data_folder, "speech_files")
        mozilla_folder = os.path.join(speech_folder, "cv-corpus-5.1-2020-06-22")

        if os.path.isdir(mozilla_folder):
            output = [[], [], []]
            files = ["train.tsv", "dev.tsv", "test.tsv"]

            lang_folders = list_dir(mozilla_folder, prefix=True)
            for lang in lang_folders:
                for idx, file in enumerate(files):
                    path = os.path.join(lang, file)
                    tmp_csv = pd.read_csv(path, sep="\t")
                    clips = os.path.join(lang, "clips")
                    clips = os.path.join(clips, "")
                    output[idx].extend(clips + tmp_csv.path[:])

            return (output[0], output[1], output[2])
        return ([], [], [])
