import os
import random
import shutil
import numpy as np
from ..utils import list_all_files, extract_from_download_cache
from .NoiseDataset import NoiseDataset
from torchvision.datasets.utils import list_dir
from pandas import DataFrame
import pandas as pd
import logging
import csv
import torchaudio
from .Downsample import Downsample


class DatasetSplit:
    def __init__(self):
        pass

    @classmethod
    def vad_balanced(cls, config):
        # directories with original data

        noise_files = NoiseDataset.getTUT_NoiseFiles(config)
        noise_files_others = NoiseDataset.getOthers_divided(config)

        speech_files_uwnu = DatasetSplit.read_UWNU(config)

        # randomly shuffle noise and speech files and split them in train,
        # validation and test set
        random.shuffle(noise_files)
        random.shuffle(speech_files_uwnu)

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

        train_speech = speech_files_uwnu[:nb_train_noise]
        dev_speech = speech_files_uwnu[nb_train_noise : nb_train_noise + nb_dev_noise]
        test_speech = speech_files_uwnu[nb_train_noise + nb_dev_noise : nb_noise_files]

        mozilla = DatasetSplit.read_mozilla(config)
        mozilla_train, mozilla_dev, mozilla_test = mozilla

        random.shuffle(mozilla_train)
        random.shuffle(mozilla_dev)
        random.shuffle(mozilla_test)

        train_speech.extend(mozilla_train[0 : len(train_other)])
        dev_speech.extend(mozilla_dev[0 : len(dev_other)])
        test_speech.extend(mozilla_test[0 : len(test_other)])

        timit_train, timit_test = DatasetSplit.read_Timit(config)

        random.shuffle(timit_train)

        nb_ttrain = int(len(timit_train) * 0.8)

        train_speech.extend(timit_train[0:nb_ttrain])
        dev_speech.extend(timit_train[nb_ttrain:])
        test_speech.extend(timit_test)

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
    def split_data(cls, config, olddata=None, split_filename=None):
        data_splits = config.get("data_split", [])
        oldsplit = {}
        if olddata is not None:
            oldsplit = olddata
        if isinstance(data_splits, str):
            if data_splits:
                data_splits = [data_splits]
            else:
                data_splits = []

        splits = ["vad_speech", "vad_balanced"]
        split_methods = [DatasetSplit.vad_speech, DatasetSplit.vad_balanced]

        for data_split in data_splits:
            logging.info("split data begins current_split: %s", data_split)
            data_folder = config["data_folder"]

            destination_dict = split_methods[splits.index(data_split)](config)

            downsample_dir = DatasetSplit.create_folder(data_folder, "downsampled")
            downsample_dir = DatasetSplit.create_folder(downsample_dir, data_split)
            speech_dir = DatasetSplit.create_folder(downsample_dir, "speech")
            noise_dir = DatasetSplit.create_folder(downsample_dir, "noise")

            dest_sr = config.get("samplingrate", 16000)

            split_file = DatasetSplit.create_filename(data_folder, split_filename)

            output = cls.file_conversion_handling(
                dest_sr, destination_dict, oldsplit, noise_dir, speech_dir
            )

            DatasetSplit.write_split(split_file, output)

    @classmethod
    def file_conversion_handling(
        cls, dest_sr, destination_dict, oldsplit, noise_dir, speech_dir
    ):
        torchaudio.set_audio_backend("sox_io")
        output = list()
        for key, value in destination_dict.items():
            for f in value:
                f_info = torchaudio.backend.sox_io_backend.info(f)
                filename = os.path.basename(f)

                old = oldsplit.pop(filename, None)
                old_orig_sr = -1
                old_down_sr = -1
                if old is not None:
                    old_orig_sr = DatasetSplit.convert_number(old.get("sr_orig"))
                    old_down_sr = DatasetSplit.convert_number(old.get("sr_down"))

                target_path = ""
                downsampled_sr = ""

                if (
                    old_orig_sr != dest_sr
                    and old_down_sr != dest_sr
                    and (dest_sr != f_info.sample_rate or f_info.num_channels != 1)
                ):
                    if "noise" in key:
                        target_path = os.path.join(noise_dir, filename)
                    elif "speech" in key:
                        target_path = os.path.join(speech_dir, filename)
                    else:
                        target_path = None

                    target_path = Downsample.downsample_file(f, target_path, dest_sr)
                    downsampled_sr = str(dest_sr)

                if oldsplit is not None and old is not None:
                    downsampled_sr = str(dest_sr)
                    if old_down_sr == dest_sr:
                        target_path = old.get("downsampled_path")
                    elif old_orig_sr == dest_sr:
                        target_path = old.get("original_path")

                output.append(
                    [filename, f, target_path, f_info.sample_rate, downsampled_sr, key]
                )
        for element in oldsplit.keys():
            tmp = oldsplit[element]
            output.append(
                [
                    tmp["filename"],
                    tmp["original_path"],
                    tmp["downsampled_path"],
                    tmp["sr_orig"],
                    tmp["sr_down"],
                    "",
                ]
            )
        return output

    @classmethod
    def convert_number(cls, text):
        output = -1
        if len(text) > 0:
            output = int(text)
        return output

    @classmethod
    def create_folder(cls, path, new_folder):
        output = os.path.join(path, new_folder)
        if not os.path.exists(output):
            os.makedirs(output)
        return output

    @classmethod
    def create_filename(cls, split_name, variants, noises, data_folder=None):
        output = (
            DatasetSplit.combine_underscore_lists(variants, noises, split_name) + ".csv"
        )

        if data_folder is not None:
            os.path.join(data_folder, output)

        return output

    @classmethod
    def create_foldername(cls, split_name, variants, noises, data_folder=None):
        output = DatasetSplit.combine_underscore_lists(variants, noises, split_name)

        if data_folder is not None:
            os.path.join(data_folder, output)

        return output

    @classmethod
    def combine_underscore_lists(cls, l1, l2, output=""):
        for variant in l1:
            output += "_" + variant

        for noise in l2:
            output += "_" + noise

        return output

    @classmethod
    def write_split(cls, output_path, data):
        header = [
            "filename",
            "original_path",
            "downsampled_path",
            "sr_orig",
            "sr_down",
            "allocation",
        ]

        with open(output_path, mode="w") as output_file:
            writer = csv.writer(output_file, delimiter=",")
            writer.writerow(header)
            writer.writerows(data)

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
    def read_Timit(cls, config):
        data_folder = config["data_folder"]
        speech_folder = os.path.join(data_folder, "timit")
        timit_data_folder = os.path.join(speech_folder, "data")

        test_folder = os.path.join(timit_data_folder, "TEST")
        train_folder = os.path.join(timit_data_folder, "TRAIN")

        if os.path.isdir(timit_data_folder):

            test = list_all_files(test_folder, ".WAV", True, ".")

            train = list_all_files(train_folder, ".WAV", True, ".")

            return (train, test)
        return ([], [])

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
