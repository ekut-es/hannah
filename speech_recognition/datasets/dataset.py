import os
import random
import re
import json
import logging
import hashlib
import sys

import librosa
import torchaudio
import numpy as np
import scipy.signal as signal
import torch
import torch.utils.data as data

from enum import Enum
from collections import defaultdict
from chainmap import ChainMap

from ..utils import list_all_files, extract_from_download_cache

msglogger = logging.getLogger()


def factor(snr, psig, pnoise):
    y = 10 ** (snr / 10)
    return np.sqrt(psig / (pnoise * y))


def load_audio(file_name, sr=16000, backend="torchaudio", res_type="kaiser_fast"):
    if backend == "librosa":
        data = librosa.core.load(file_name, sr=sr, res_type=res_type)
    elif backend == "torchaudio":
        torchaudio.set_audio_backend("sox")
        try:
            data, samplingrate = torchaudio.load(file_name)
        except:
            msglogger.warning(
                "Could not load %s with default backend trying sndfile", str(file_name)
            )
            torchaudio.set_audio_backend("soundfile")
            data, samplingrate = torchaudio.load(file_name)
        if samplingrate != sr:
            data = torchaudio.transforms.Resample(samplingrate, sr).forward(data)
        data = data.numpy()
    else:
        raise Exception(f"Unknown backend name {backend}")

    return data


class DatasetType(Enum):
    """ The type of a dataset partition e.g. train, dev, test """

    TRAIN = 0
    DEV = 1
    TEST = 2


class SpeechDataset(data.Dataset):
    """ Base Class for speech datasets """

    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    LABEL_NOISE = "__noise__"

    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())

        config["bg_noise_files"] = list(
            filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", []))
        )
        self.samplingrate = config["samplingrate"]
        self.bg_noise_audio = [
            load_audio(file, sr=self.samplingrate)[0]
            for file in config["bg_noise_files"]
        ]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]

        self.extract = config["extract"]
        self.unknown_class = 1
        self.silence_class = 0
        n_unk = len(list(filter(lambda x: x == self.unknown_class, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.max_feature = 0
        self.train_snr_low = config["train_snr_low"]
        self.train_snr_high = config["train_snr_high"]
        self.test_snr = config["test_snr"]
        self.channels = 1  # FIXME: add config option

    def _timeshift_audio(self, data):
        """Shifts data by a random amount of ms given by parameter timeshift_ms"""
        shift = (self.samplingrate * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[: len(data) - a] if a else data[b:]

    def _extract_random_range(self, data, in_len):
        """Extract random part of the sample with length self.input_length"""
        if len(data) <= in_len:
            return (0, len(data))
        elif (int(len(data) * 0.8) - 1) < in_len:
            rand_end = len(data) - in_len
            cutstart = np.random.randint(0, rand_end)
            return (cutstart, cutstart + in_len)
        else:
            max_length = int(len(data) * 0.8)
            max_length = max_length - in_len
            cutstart = np.random.randint(0, max_length)
            cutstart = cutstart + int(len(data) * 0.1)
            return (cutstart, cutstart + in_len)

    def _extract_front_range(self, data, in_len):
        """Extract front part of the sample with length self.input_length"""
        if len(data) <= in_len:
            return (0, len(data))
        else:
            return (0, in_len)

    def _extract_loudest_range(self, data, in_len):
        """Extract the loudest part of the sample with length self.input_length"""
        if len(data) <= in_len:
            return (0, len(data))

        amps = np.abs(data)
        f = np.ones(in_len)

        correlation = signal.correlate(amps, f)

        window_start = np.argmax(correlation)
        window_start = max(0, window_start)

        return (window_start, window_start + in_len)

    def preprocess(self, example, silence=False, label=0):
        """ Run preprocessing and feature extraction """

        if silence:
            example = "__silence__"

        in_len = self.input_length

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            data = load_audio(example, sr=self.samplingrate)[0]

            extract_index = (0, len(data))

            if self.extract == "loudest":
                extract_index = self._extract_loudest_range(data, in_len)
            elif self.extract == "trim_border":
                extract_index = self._extract_random_range(data, in_len)
            elif self.extract == "front":
                extract_index = self._extract_front_range(data, in_len)

            data = self._timeshift_audio(data)
            data = data[extract_index[0] : extract_index[1]]

            data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
            data = data[0:in_len]

        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - data.shape[0] - 1)
            bg_noise = bg_noise[a : a + data.shape[0]]

        else:
            # Same formula as used for google kws white noise
            bg_noise = np.random.normal(0, 1, data.shape[0]) / 3
            bg_noise = np.float32(bg_noise)

        if self.set_type == DatasetType.TEST:
            snr = self.test_snr
        else:
            snr = random.uniform(self.train_snr_low, self.train_snr_high)

        psig = np.sum(data * data) / len(data)
        pnoise = np.sum(bg_noise * bg_noise) / len(bg_noise)
        if psig == 0.0:
            data = bg_noise
        else:
            if snr != float("inf"):
                f = factor(snr, psig, pnoise)
                data = data + f * bg_noise

                if np.amax(np.absolute(data)) > 1:
                    data = data / np.amax(np.absolute(data))

        data = torch.from_numpy(data)
        data = torch.unsqueeze(data, 0)

        return data

    def get_class(self, index):
        label = []
        if index >= len(self.audio_labels):
            label = [self.silence_class]
        else:
            label = [self.audio_labels[index]]

        return label

    def get_classes(self):
        labels = []
        for i in range(len(self)):
            labels.append(self.get_class(i))

        return labels

    def get_class_nums(self):
        classcounter = defaultdict(int)
        for n in self.get_classes():
            for c in n:
                classcounter[c] += 1

        return classcounter

    def __getitem__(self, index):

        label = torch.Tensor(self.get_class(index))
        label = label.long()

        if index >= len(self.audio_labels):
            data = self.preprocess(None, silence=True)
        else:
            data = self.preprocess(self.audio_files[index])

        return data, data.shape[0], label, label.shape[0]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence


class SpeechCommandsDataset(SpeechDataset):
    """This class implements reading and preprocessing of speech commands like
    dataset"""

    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)

        self.label_names = {0: self.LABEL_SILENCE, 1: self.LABEL_UNKNOWN}
        for i, word in enumerate(config["wanted_words"]):
            self.label_names[i + 2] = word

    @classmethod
    def splits(cls, config):
        msglogger = logging.getLogger()

        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(config["data_folder"], "speech_commands_v0.02")
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        use_default_split = config["use_default_split"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE: 0, cls.LABEL_UNKNOWN: 1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        test_files = set()
        dev_files = set()
        if use_default_split:
            with open(os.path.join(folder, "testing_list.txt")) as testing_list:
                for line in testing_list.readlines():
                    line = line.strip()
                    test_files.add(os.path.join(folder, line))

            with open(os.path.join(folder, "validation_list.txt")) as validation_list:
                for line in validation_list.readlines():
                    line = line.strip()
                    dev_files.add(os.path.join(folder, line))

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue

                if use_default_split:
                    if wav_name in dev_files:
                        tag = DatasetType.DEV
                    elif wav_name in test_files:
                        tag = DatasetType.TEST
                    else:
                        tag = DatasetType.TRAIN
                else:
                    if config["group_speakers_by_id"]:
                        hashname = re.sub(r"_nohash_.*$", "", filename)
                    else:
                        hashname = filename
                    max_no_wavs = 2 ** 27 - 1
                    bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                    bucket = (bucket % (max_no_wavs + 1)) * (100.0 / max_no_wavs)
                    if bucket < dev_pct:
                        tag = DatasetType.DEV
                    elif bucket < test_pct + dev_pct:
                        tag = DatasetType.TEST
                    else:
                        tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        datasets = (
            cls(sets[0], DatasetType.TRAIN, train_cfg),
            cls(sets[1], DatasetType.DEV, test_cfg),
            cls(sets[2], DatasetType.TEST, test_cfg),
        )
        return datasets

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]

        if len(downloadfolder_tmp) == 0:
            downloadfolder_tmp = os.path.join(
                sys.argv[0].replace("speech_recognition/train.py", ""),
                "datasets/downloads",
            )

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".tar.gz")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        variants = config["variants"]

        target_folder = os.path.join(data_folder, "speech_commands_v0.02")

        # download speech_commands dataset
        filename = "speech_commands_v0.02.tar.gz"

        if "speech_command" in variants:
            extract_from_download_cache(
                filename,
                "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                cached_files,
                os.path.join(downloadfolder_tmp, "speech_commands"),
                target_folder,
                clear_download,
            )


class SpeechHotwordDataset(SpeechDataset):
    """Dataset Class for Hotword dataset e.g. Hey Snips!"""

    LABEL_HOTWORD = "hotword"
    LABEL_EPS = "eps"

    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)

        self.label_names = {
            0: self.LABEL_SILENCE,
            1: self.LABEL_UNKNOWN,
            2: self.LABEL_HOTWORD,
        }

    @staticmethod
    def default_config():
        config = SpeechDataset.default_config()
        config["n_labels"].default = 3

        # Splits the dataset in 1/3
        config["silence_prob"].default = 1.0
        config["unknown_prob"].default = 1.0
        return config

    @classmethod
    def splits(cls, config):
        """Splits the dataset in training, devlopment and test set and returns
        the three sets as List"""

        msglogger = logging.getLogger()

        folder = config["data_folder"]
        folder = os.path.join(folder, "hey_snips_research_6k_en_train_eval_clean_ter")

        descriptions = ["train.json", "dev.json", "test.json"]
        dataset_types = [DatasetType.TRAIN, DatasetType.DEV, DatasetType.TEST]
        datasets = [{}, {}, {}]

        for num, desc in enumerate(descriptions):
            descs = os.path.join(folder, desc)
            with open(descs) as f:
                descs = json.load(f)

                unknown_files = []
                hotword_files = []

                for desc in descs:
                    is_hotword = desc["is_hotword"]
                    wav_file = os.path.join(folder, desc["audio_file_path"])

                    if is_hotword:
                        hotword_files.append(wav_file)
                    else:
                        unknown_files.append(wav_file)

            unknown_prob = config["unknown_prob"]
            num_hotwords = len(hotword_files)
            num_unknowns = int(num_hotwords * unknown_prob)
            random.shuffle(unknown_files)
            label_unknown = 1
            label_hotword = 2

            datasets[num].update(
                {u: label_unknown for u in unknown_files[:num_unknowns]}
            )
            datasets[num].update({h: label_hotword for h in hotword_files})

        res_datasets = (
            cls(datasets[0], DatasetType.TRAIN, config),
            cls(datasets[1], DatasetType.DEV, config),
            cls(datasets[2], DatasetType.TEST, config),
        )

        return res_datasets

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]
        variants = config["variants"]

        if len(downloadfolder_tmp) == 0:
            downloadfolder_tmp = os.path.join(
                sys.argv[0].replace("speech_recognition/train.py", ""),
                "datasets/downloads",
            )

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".tar.gz")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        snips_target = os.path.join(
            data_folder, "hey_snips_research_6k_en_train_eval_clean_ter"
        )
        snips_filename = "hey_snips_kws_4.0.tar.gz"
        url = "https://atreus.informatik.uni-tuebingen.de/seafile/f/2e950ff3abbc4c46828e/?dl=1"

        if "snips" in variants:
            extract_from_download_cache(
                snips_filename,
                url,
                cached_files,
                downloadfolder_tmp,
                snips_target,
                snips_target,
            )


class VadDataset(SpeechDataset):
    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)

        self.label_names = {0: "noise", 1: "speech"}

    @classmethod
    def splits(cls, config):
        """Splits the dataset in training, devlopment and test set and returns
        the three sets as List"""

        msglogger = logging.getLogger()

        folder = config["data_folder"]
        folder = os.path.join(folder, "vad_balanced")

        descriptions = ["train", "dev", "test"]
        dataset_types = [DatasetType.TRAIN, DatasetType.DEV, DatasetType.TEST]
        datasets = [{}, {}, {}]
        configs = [{}, {}, {}]

        for num, desc in enumerate(descriptions):

            descs_noise = os.path.join(folder, desc, "noise")
            descs_speech = os.path.join(folder, desc, "speech")
            descs_bg = os.path.join(folder, desc, "background_noise")

            noise_files = [
                os.path.join(descs_noise, f)
                for f in os.listdir(descs_noise)
                if os.path.isfile(os.path.join(descs_noise, f))
            ]
            speech_files = [
                os.path.join(descs_speech, f)
                for f in os.listdir(descs_speech)
                if os.path.isfile(os.path.join(descs_speech, f))
            ]
            bg_noise_files = [
                os.path.join(descs_bg, f)
                for f in os.listdir(descs_bg)
                if os.path.isfile(os.path.join(descs_bg, f))
            ]

            random.shuffle(noise_files)
            random.shuffle(speech_files)
            label_noise = 0
            label_speech = 1

            datasets[num].update({n: label_noise for n in noise_files})
            datasets[num].update({s: label_speech for s in speech_files})
            configs[num].update(ChainMap(dict(bg_noise_files=bg_noise_files), config))

        res_datasets = (
            cls(datasets[0], DatasetType.TRAIN, configs[0]),
            cls(datasets[1], DatasetType.DEV, configs[1]),
            cls(datasets[2], DatasetType.TEST, configs[2]),
        )

        return res_datasets

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]

        if len(downloadfolder_tmp) == 0:
            downloadfolder_tmp = os.path.join(
                sys.argv[0].replace("speech_recognition/train.py", ""),
                "datasets/downloads",
            )

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".tar.gz")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        speechdir = os.path.join(data_folder, "speech_files")

        if not os.path.isdir(speechdir):
            os.makedirs(speechdir)

        variants = config["variants"]

        # download mozilla dataset
        target_cache_mozilla = os.path.join(downloadfolder_tmp, "mozilla")
        mozilla_supertarget = os.path.join(speechdir, "cv-corpus-5.1-2020-06-22")

        if "en" in variants:
            filename = "en.tar.gz"
            target_test_folder = os.path.join(mozilla_supertarget, "en")
            extract_from_download_cache(
                filename,
                "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz",
                cached_files,
                target_cache_mozilla,
                speechdir,
                target_test_folder=target_test_folder,
                clear_download=clear_download,
            )

        mozilla_lang = ["de", "fr", "es", "it"]
        for name in mozilla_lang:
            filename = name + ".tar.gz"
            target_test_folder = os.path.join(mozilla_supertarget, name)
            url = (
                "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/"
                + filename
            )

            if name in variants:
                extract_from_download_cache(
                    filename,
                    url,
                    cached_files,
                    target_cache_mozilla,
                    speechdir,
                    target_test_folder,
                    clear_download,
                )

        # download UWNU dataset
        if "UWNU" in variants:
            filename = "uwnu-v2.tar.gz"
            target_test_folder = os.path.join(speechdir, "uwnu-v2")
            url = "https://atreus.informatik.uni-tuebingen.de/seafile/f/bfc1be836c7a4e339215/?dl=1"
            target_cache = os.path.join(downloadfolder_tmp, "UWNU")

            extract_from_download_cache(
                filename,
                url,
                cached_files,
                target_cache,
                speechdir,
                target_test_folder,
                clear_download,
            )


class KeyWordDataset(SpeechDataset):
    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)

        self.label_names = {0: self.LABEL_SILENCE, 1: self.LABEL_UNKNOWN}
        l_noise = 2
        for i, word in enumerate(config["wanted_words"]):
            self.label_names[i + 2] = word
            l_noise = l_noise + 1
        self.label_names[l_noise] = self.LABEL_NOISE

    @classmethod
    def splits(cls, config):
        msglogger = logging.getLogger()

        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        use_default_split = config["use_default_split"]

        words = {}
        l_noise = 2
        for i, word in enumerate(wanted_words):
            words.update({word: i + 2})
            l_noise = l_noise + 1
        words.update(
            {cls.LABEL_SILENCE: 0, cls.LABEL_UNKNOWN: 1, cls.LABEL_NOISE: l_noise}
        )
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        test_files = set()
        dev_files = set()
        if use_default_split:
            with open(os.path.join(folder, "testing_list.txt")) as testing_list:
                for line in testing_list.readlines():
                    line = line.strip()
                    test_files.add(os.path.join(folder, line))

            with open(os.path.join(folder, "validation_list.txt")) as validation_list:
                for line in validation_list.readlines():
                    line = line.strip()
                    dev_files.add(os.path.join(folder, line))

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_chunks_":
                label = words[cls.LABEL_NOISE]
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue

                if use_default_split:
                    if wav_name in dev_files:
                        tag = DatasetType.DEV
                    elif wav_name in test_files:
                        tag = DatasetType.TEST
                    else:
                        tag = DatasetType.TRAIN
                else:
                    if config["group_speakers_by_id"]:
                        hashname = re.sub(r"_nohash_.*$", "", filename)
                    else:
                        hashname = filename
                    max_no_wavs = 2 ** 27 - 1
                    bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                    bucket = (bucket % (max_no_wavs + 1)) * (100.0 / max_no_wavs)
                    if bucket < dev_pct:
                        tag = DatasetType.DEV
                    elif bucket < test_pct + dev_pct:
                        tag = DatasetType.TEST
                    else:
                        tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        datasets = (
            cls(sets[0], DatasetType.TRAIN, config),
            cls(sets[1], DatasetType.DEV, config),
            cls(sets[2], DatasetType.TEST, config),
        )
        return datasets

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]

        if len(downloadfolder_tmp) == 0:
            downloadfolder_tmp = os.path.join(
                sys.argv[0].replace("speech_recognition/train.py", ""),
                "datasets/downloads",
            )

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".tar.gz")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        variants = config["variants"]

        target_folder = os.path.join(data_folder, "speech_commands_v0.02")

        # download speech_commands dataset
        filename = "speech_commands_v0.02.tar.gz"

        if "speech_command" in variants:
            extract_from_download_cache(
                filename,
                "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                cached_files,
                os.path.join(downloadfolder_tmp, "speech_commands"),
                target_folder,
                clear_download,
            )


def ctc_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, src_length, trg_seq, trg_length).
            - src_seq: torch tensor of shape (x,?); variable length.
            - src length: torch tenso of shape 1x1
            - trg_seq: torch tensor of shape (?); variable length.
            - trg_length: torch_tensor of shape (1x1)
    Returns: tuple of four torch tensors
        src_seqs: torch tensor of shape (batch_size, x, padded_length).
        src_lengths: torch_tensor of shape (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, x, padded_length).
        trg_lengths: torch tensor of shape (batch_size); valid length for each padded target sequence.
    """

    def merge(sequences):
        temporal_dimension = 0
        lengths = [seq.shape[-1] for seq in sequences]
        max_length = max(lengths)

        padded_seqs = []

        for item in sequences:
            padded = torch.nn.functional.pad(
                input=item,
                pad=(0, max_length - item.shape[-1]),
                mode="constant",
                value=0,
            )
            padded_seqs.append(padded)

        return padded_seqs, lengths

    # seperate source and target sequences
    src_seqs, src_lengths, trg_seqs, trg_lengths = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return (
        torch.stack(src_seqs),
        torch.Tensor(src_lengths),
        torch.stack(trg_seqs),
        torch.Tensor(trg_lengths),
    )
