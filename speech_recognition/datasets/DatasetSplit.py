import os
import random
import shutil
import numpy as np
from ..utils import list_all_files, extract_from_download_cache


class DatasetSplit:
    def __init__(self):
        pass

    @classmethod
    def load_vad_default(cls, speechdir, noisedir):
        noise_files = list_all_files(noisedir, ".wav", True, ".")
        noise_files.extend(list_all_files(noisedir, ".mp3", True, "."))

        speech_files = list_all_files(speechdir, ".wav", True, ".")
        speech_files.extend(list_all_files(speechdir, ".mp3", True, "."))

        return (speech_files, noise_files)

    @classmethod
    def vad(cls, config):
        # directories with original data
        data_folder = config["data_folder"]
        noise_dir = os.path.join(data_folder, "noise_files")
        speech_dir = os.path.join(data_folder, "speech_files")

        # list all noise  and speech files
        speech_files, noise_files = DatasetSplit.load_vad_default(speech_dir, noise_dir)

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

        train_speech = speech_files[:nb_train_speech]
        dev_speech = speech_files[nb_train_speech : nb_train_speech + nb_dev_speech]
        test_speech = speech_files[nb_train_speech + nb_dev_speech :]

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

        speech_files, noise_files = DatasetSplit.load_vad_default(speech_dir, noise_dir)

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
    def getrennt(cls, config):
        # directories with original data
        data_folder = config["data_folder"]
        noise_dir = os.path.join(data_folder, "noise_files")
        speech_dir = os.path.join(data_folder, "speech_files")

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
        noise_files = list_all_files(noise_dir, ".wav", True)
        noise_files.extend(list_all_files(noise_dir, ".mp3", True))

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
