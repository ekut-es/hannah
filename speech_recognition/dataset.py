from enum import Enum
import hashlib
import math
import os
import random
import re
import json
import logging

from chainmap import ChainMap
import librosa
import numpy as np
import torch
import torch.utils.data as data

from .config import ConfigOption
from .process_audio import preprocess_audio, calculate_feature_shape


class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())

        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.samplingrate = config["samplingrate"]
        self.bg_noise_audio = [librosa.core.load(file, sr=self.samplingrate)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self.extract_loudest = config["extract_loudest"]
        self.loss_function = config["loss"]
        self.dct_filters = librosa.filters.dct(config["n_mfcc"], config["n_mels"])
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        self.cache_prob = config["cache_prob"]
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.features = config['features']
        self.n_mfcc = config["n_mfcc"]
        self.n_mels = config["n_mels"]
        self.stride_ms = config["stride_ms"] 
        self.window_ms = config["window_ms"]
        self.freq_min = config["freq_min"]
        self.freq_max = config["freq_max"]
        
        self.height, self.width = calculate_feature_shape(self.input_length,
                                                          features=self.features, 
                                                          samplingrate=self.samplingrate,
                                                          n_mels=self.n_mels,
                                                          n_mfcc=self.n_mfcc, 
                                                          stride_ms=self.stride_ms,
                                                          window_ms=self.window_ms)
        
    @staticmethod
    def default_config():
        config = {}
        
        #Input Description
        config["wanted_words"]         = ConfigOption(category="Input Config",
                                                      default=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]),
        config["data_folder"]          = ConfigOption(category="Input Config",
                                                      default="datasets/speech_commands_v0.02/")
        config["samplingrate"]         = ConfigOption(category="Input Config",
                                                      default=16000),
        config["input_length"]         = ConfigOption(category="Input Config",
                                                      default=16000),
        config["extract_loudest"]      = ConfigOption(category="Input Config",
                                                      default=True),
        config["timeshift_ms"]         = ConfigOption(category="Input Config",
                                                      default=100)
        config["use_default_split"]    = ConfigOption(category="Input Config",
                                                      default=False)
        config["group_speakers_by_id"] = ConfigOption(category="Input Config",
                                                      default=True)
        config["silence_prob"]         = ConfigOption(category="Input Config",
                                                      default=0.1)
        config["noise_prob"]           = ConfigOption(category="Input Config",
                                                      default=0.8)
        config["unknown_prob"]         = ConfigOption(category="Input Config",
                                                      default=0.1)
        config["train_pct"]            = ConfigOption(category="Input Config",
                                                      default=80)
        config["dev_pct"]              = ConfigOption(category="Input Config",
                                                      default=10)
        config["test_pct"]             = ConfigOption(category="Input Config",
                                                      default=10)
        config["trim"]                 = ConfigOption(category="Input Config",
                                                      default=True)
        config["loss"]                 = ConfigOption(category="Input Config",
                                                      default="cross_entropy")
        
        # Feature extraction
        config["features"]  = ConfigOption(category="Feature Config",
                                           default="mel")
        config["n_mfcc"]    = ConfigOption(category="Feature Config",
                                           default=40)
        config["n_mels"]    = ConfigOption(category="Feature Config",
                                           default=40)
        config["stride_ms"] = ConfigOption(category="Feature Config",
                                           default=10)
        config["window_ms"] = ConfigOption(category="Feature Config",
                                           default=30)
        config["freq_min"]  = ConfigOption(category="Feature Config",
                                           default=20)
        config["freq_max"]  = ConfigOption(category="Feature Config",
                                           default=4000)
    
        # Cache config
        config["cache_size"] = ConfigOption(category="Cache Config",
                                            default=200000,
                                            desc="Size of the caches for preprocessed and raw data")
        config["cache_prob"] = ConfigOption(category="Cache Config",
                                            default=0.8,
                                            desc="Probabilty of using a cached sample during training")

        return config

    def _timeshift_audio(self, data):
        shift = (self.samplingrate * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def _extract_loudest_range(self, data, in_len):
        """Extract the loudest part of the sample with length self.input_lenght"""
        if len(data) <= in_len:
            return (0, len(data))
        
        amps = np.abs(data)
        f = np.ones(in_len)

        correlation = np.correlate(amps, f)
        
        window_start = np.argmax(correlation)
        window_start = max(0, window_start)

        return (window_start, window_start+in_len)
        
        
    def preprocess(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() <= self.cache_prob:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass

        if self.loss_function == "ctc":
            in_len = 16000 * 4
        else:
            in_len = self.input_length

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            data = self._file_cache.get(example)

            if data is None:
                data = librosa.core.load(example, sr=self.samplingrate)[0]

                extract_index = (0, len(data))
                
                if self.extract_loudest:
                    extract_index = self._extract_loudest_range(data, in_len)
                 
                data = self._timeshift_audio(data)
                data = data[extract_index[0]:extract_index[1]]
                 
                data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
                data = data[0:in_len]

                    
            self._file_cache[example] = data

        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - data.shape[0] - 1)
            bg_noise = bg_noise[a:a + data.shape[0]]
        else:
            bg_noise = np.zeros(data.shape[0])

            
        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1 
            data = np.clip(a * bg_noise + data, -1, 1)
        data = torch.from_numpy(preprocess_audio(data,
                                                 features = self.features,
                                                 samplingrate = self.samplingrate,
                                                 n_mels = self.n_mels,
                                                 n_mfcc = self.n_mfcc,
                                                 dct_filters = self.dct_filters,
                                                 freq_min = self.freq_min,
                                                 freq_max = self.freq_max,
                                                 window_ms = self.window_ms,
                                                 stride_ms = self.stride_ms))
        
        if self.loss_function != "ctc":
            assert data.shape[0] == self.height
            assert data.shape[1] == self.width

        self._audio_cache[example] = data

        return data


    def __getitem__(self, index):

        a = 0
        if self.loss_function == "ctc":
            a = 1
        
        if index >= len(self.audio_labels):
            return self.preprocess(None, silence=True), 0 + a
        return self.preprocess(self.audio_files[index]), self.audio_labels[index] + a

    def __len__(self):
        return len(self.audio_labels) + self.n_silence


class SpeechCommandsDataset(SpeechDataset):

    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)
    
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
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
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
                    max_no_wavs = 2**27 - 1
                    bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                    bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
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
    
        msglogger.info("Dataset config:")
        msglogger.info("  train: %d", len(sets[0]))
        msglogger.info("  dev:   %d", len(sets[1]))
        msglogger.info("  test:  %d", len(sets[2]))
        msglogger.info("  total: %d", len(sets[0])+len(sets[1])+len(sets[2]))
        msglogger.info("")
            
        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg),
                    cls(sets[1], DatasetType.DEV, test_cfg),
                    cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets


class SpeechHotwordDataset(SpeechDataset):

    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)
    
    
    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]

        descriptions = ["train.json", "dev.json", "test.json"]
        dataset_types = [DatasetType.TRAIN, DatasetType.DEV, DatasetType.TEST]
        datasets=[{}, {}, {}]

        for num, desc in enumerate(descriptions):
            descs = os.path.join(folder, desc)
            with open(descs) as f:
                descs = json.load(f)

                for desc in descs:
                    is_hotword = desc["is_hotword"]
                    wav_file = os.path.join(folder, desc["audio_file_path"])

                    label = 2 if is_hotword else 1
                    
                    datasets[num][wav_file] = label
                    
                    
        res_datasets = (cls(datasets[0], DatasetType.TRAIN, config),
                        cls(datasets[1], DatasetType.DEV, config),
                        cls(datasets[2], DatasetType.TEST, config))

        return res_datasets


def find_dataset(name):
    if name == "keywords":
        return SpeechCommandsDataset
    elif name == "hotword":
        return SpeechHotwordDataset

    raise Exception("Could not find dataset type: {}".format(name))
