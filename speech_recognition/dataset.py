from enum import Enum
import hashlib
import math
import os
import random
import re

from chainmap import ChainMap
import librosa
import numpy as np
import torch
import torch.utils.data as data

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
        self.dct_filters = librosa.filters.dct(config["n_dct"], config["n_mels"])
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        self.cache_prob = config["cache_prob"]
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.features = config['features']
        self.n_dct_filters = config["n_dct"]
        self.n_mels = config["n_mels"]
        self.stride_ms = config["stride_ms"] 
        self.window_ms = config["window_ms"]
        self.freq_min = config["freq_min"]
        self.freq_max = config["freq_max"]
        
        self.height, self.width = calculate_feature_shape(self.input_length, self.features, 
                                                          self.samplingrate, self.n_dct_filters, 
                                                          self.stride_ms)
        
    @staticmethod
    def default_config():
        config = {}
        
        #Input Description
        config["wanted_words"] = ["yes", "no", "up", "down", "left",
                                  "right", "on", "off", "stop", "go"]
        config["data_folder"] = "datasets/speech_commands_v0.02/"
        config["samplingrate"] = 16000
        config["input_length"] = 16000
        config["timeshift_ms"] = 100

        # Input Enhancement
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["unknown_prob"] = 0.1
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10

        # Feature extraction
        config["features"] = "mel"
        config["n_dct"] = 40
        config["n_mels"] = 40
        config["stride_ms"] = 10
        config["window_ms"] = 30
        config["freq_min"] = 20
        config["freq_max"] = 4000
    
        # Cache config
        config["cache_size"] = 200000
        config["cache_prob"] = 0.7

        return config

    def _timeshift_audio(self, data):
        shift = (self.samplingrate * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def preprocess(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() <= self.cache_prob:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = librosa.core.load(example, sr=self.samplingrate)[0] if file_data is None else file_data
            self._file_cache[example] = data
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        data = data[0:in_len]
        
        if self.set_type == DatasetType.TRAIN:
            data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)
        data = torch.from_numpy(preprocess_audio(data,
                                                 features = self.features,
                                                 samplingrate = self.samplingrate,
                                                 n_mels = self.n_mels,
                                                 dct_filters = self.dct_filters,
                                                 freq_min = self.freq_min,
                                                 freq_max = self.freq_max,
                                                 window_ms = self.window_ms,
                                                 stride_ms = self.stride_ms))

        assert data.shape[0]  == self.height
        assert data.shape[1] == self.width

        self._audio_cache[example] = data
        return data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

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
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
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

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg),
                    cls(sets[1], DatasetType.DEV, test_cfg),
                    cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.preprocess(None, silence=True), 0
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence
