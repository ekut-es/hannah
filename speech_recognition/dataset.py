from enum import Enum
import hashlib
import math
import os
import random
import re
import json
import logging
from collections import defaultdict

from chainmap import ChainMap
import librosa
import torchaudio
import numpy as np
import scipy.signal as signal
import torch
import torch.utils.data as data
import hashlib
import pickle

from .config import ConfigOption
from .process_audio import preprocess_audio, calculate_feature_shape


def factor(snr, psig, pnoise):
    y = 10 ** (snr / 10)
    return np.sqrt(psig / (pnoise * y))


def load_audio(file_name, sr=16000, backend="torchaudio", res_type="kaiser_fast"):
    if backend == "librosa":
        data = librosa.core.load(file_name, sr=sr, res_type=res_type)
    elif backend == "torchaudio":
        torchaudio.set_audio_backend("sox")
        data, samplingrate = torchaudio.load(file_name)
        data = data.numpy()
        if samplingrate != sr:
            data = librosa.resample(data, sampling_sr, sr, res_type=res_type)
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
        self.extract_loudest = config["extract_loudest"]
        self.dct_filters = librosa.filters.dct(config["n_mfcc"], config["n_mels"])
        self.randomstates = dict()
        self.unknown_class = 1
        self.silence_class = 0
        n_unk = len(list(filter(lambda x: x == self.unknown_class, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.features = config["features"]
        self.n_mfcc = config["n_mfcc"]
        self.n_mels = config["n_mels"]
        self.stride_ms = config["stride_ms"]
        self.window_ms = config["window_ms"]
        self.freq_min = config["freq_min"]
        self.freq_max = config["freq_max"]
        self.normalize_bits = config["normalize_bits"]
        self.normalize_max = config["normalize_max"]
        self.max_feature = 0
        self.train_snr_low = config["train_snr_low"]
        self.train_snr_high = config["train_snr_high"]
        self.test_snr = config["test_snr"]

        self.height, self.width = calculate_feature_shape(
            self.input_length,
            features=self.features,
            samplingrate=self.samplingrate,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            stride_ms=self.stride_ms,
            window_ms=self.window_ms,
        )

    @staticmethod
    def default_config():
        """ Returns the default configuration for the Dataset and
        Feature extraction"""
        config = {}

        # Input Description
        config["wanted_words"] = ConfigOption(
            category="Input Config",
            default=[
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
            ],
        )
        config["n_labels"] = ConfigOption(category="Input Config", default=12)
        config["data_folder"] = ConfigOption(
            category="Input Config", default="datasets/speech_commands_v0.02/"
        )
        config["samplingrate"] = ConfigOption(category="Input Config", default=16000)
        config["input_length"] = ConfigOption(category="Input Config", default=16000)
        config["extract_loudest"] = ConfigOption(category="Input Config", default=True)
        config["timeshift_ms"] = ConfigOption(category="Input Config", default=100)
        config["use_default_split"] = ConfigOption(
            category="Input Config", default=False
        )
        config["group_speakers_by_id"] = ConfigOption(
            category="Input Config", default=True
        )
        config["silence_prob"] = ConfigOption(category="Input Config", default=0.1)
        config["unknown_prob"] = ConfigOption(category="Input Config", default=0.1)
        config["train_pct"] = ConfigOption(category="Input Config", default=80)
        config["dev_pct"] = ConfigOption(category="Input Config", default=10)
        config["test_pct"] = ConfigOption(category="Input Config", default=10)
        config["train_snr_low"] = ConfigOption(category="Input Config", default=0.0)
        config["train_snr_high"] = ConfigOption(category="Input Config", default=20.0)
        config["test_snr"] = ConfigOption(
            category="Input Config", desc="SNR used during test", default=float("inf")
        )

        # Feature extraction
        config["features"] = ConfigOption(
            category="Feature Config",
            choices=["mel", "mfcc", "melspec", "spectrogram", "raw"],
            default="mel",
        )
        config["n_mfcc"] = ConfigOption(category="Feature Config", default=40)
        config["n_mels"] = ConfigOption(category="Feature Config", default=40)
        config["stride_ms"] = ConfigOption(category="Feature Config", default=10)
        config["window_ms"] = ConfigOption(category="Feature Config", default=30)
        config["freq_min"] = ConfigOption(category="Feature Config", default=20)
        config["freq_max"] = ConfigOption(category="Feature Config", default=4000)

        config["normalize_bits"] = ConfigOption(
            category="Feature Config",
            desc="Normalize features to n bits 0 means no normalization",
            default=0,
        )
        config["normalize_max"] = ConfigOption(
            category="Feature Config",
            desc="Divide features by this value before normalization",
            default=256,
        )

        return config

    def _timeshift_audio(self, data):
        """Shifts data by a random amount of ms given by parameter timeshift_ms"""
        shift = (self.samplingrate * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[: len(data) - a] if a else data[b:]

    def _extract_loudest_range(self, data, in_len):
        """Extract the loudest part of the sample with length self.input_lenght"""
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

            if self.extract_loudest:
                extract_index = self._extract_loudest_range(data, in_len)

            data = self._timeshift_audio(data)
            data = data[extract_index[0] : extract_index[1]]

        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - data.shape[0] - 1)
            bg_noise = bg_noise[a : a + data.shape[0]]

        else:
            bg_noise = np.zeros(data.shape[0])

        if self.set_type == DatasetType.TEST:
            snr = self.test_snr
        else:
            snr = random.uniform(self.train_snr_low, self.train_snr_high)

        if snr != float("inf"):
            psig = np.sum(data * data) / len(data)
            pnoise = np.sum(bg_noise * bg_noise) / len(bg_noise)
            f = factor(snr, psig, pnoise)
            data = data + f * bg_noise

            if np.amax(np.absolute(data)) > 1:
                data = data / np.amax(np.absolute(data))

        data = preprocess_audio(
            data,
            features=self.features,
            samplingrate=self.samplingrate,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            dct_filters=self.dct_filters,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            window_ms=self.window_ms,
            stride_ms=self.stride_ms,
        )

        data = torch.from_numpy(data)

        if self.normalize_bits > 0:
            normalize_factor = 2.0 ** (self.normalize_bits - 1)

            data = data / self.normalize_max * normalize_factor
            data = data.round()
            data = data / normalize_factor
            data = data.clamp(-1.0, 1.0 - 1.0 / normalize_factor)

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

        return data, data.shape[1], label, label.shape[0]

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

        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
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


def find_dataset(name):
    """Returns the appropriate class for reading a dataset of type name:

       Parameters
       ----------
       name : str
           The name of the dataset type

            - keywords = Google Speech Commands like  dataset
            - hotword = Hey Snips! like dataset

       Returns
"""
    if name == "keywords":
        return SpeechCommandsDataset
    elif name == "hotword":
        return SpeechHotwordDataset
    elif name == "vad":
        return VadDataset
    elif name == "keywords_and_noise":
        return KeyWordDataset

    raise Exception("Could not find dataset type: {}".format(name))


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
