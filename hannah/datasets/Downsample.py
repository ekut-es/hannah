import logging
import os

import numpy as np
import torchaudio
from torchvision.datasets.utils import list_dir, list_files

from ..utils import list_all_files

logger = logging.getLogger(__name__)


class Downsample:
    def __init__(self):
        pass

    @classmethod
    def downsample_file(cls, sourcepath, targetpath, target_sr):
        assert targetpath is not None
        torchaudio.set_audio_backend("sox_io")
        data, sr = torchaudio.load(sourcepath)
        f_info = torchaudio.backend.sox_io_backend.info(sourcepath)
        changed = False
        if target_sr != sr:
            data = torchaudio.transforms.Resample(sr, target_sr).forward(data)
            changed = True
        elif f_info.num_channels != 1:
            changed = True

        if targetpath.endswith("mp3"):
            targetpath = targetpath.replace(".mp3", ".wav")
            changed = True
        if targetpath.endswith("MP3"):
            targetpath = targetpath.replace(".MP3", ".wav")
            changed = True
        if changed:
            torchaudio.save(targetpath, data[0].unsqueeze(0), target_sr)
            return targetpath

        return None

    @classmethod
    def downsample(cls, config):
        if "downsample" not in config:
            return

        samplerate = config["downsample"]
        if samplerate > 0:
            logger.info("downsample data begins")
            config["downsample"] = 0
            downsample_folder = ["train", "dev", "test"]
            torchaudio.set_audio_backend("sox_io")

            if len(config["data_split"]) != 0:
                downsample_folder = os.path.join(
                    config["data_folder"], config["data_split"]
                )
                files = list_all_files(downsample_folder, ".mp3", True)
                files.extend(list_all_files(downsample_folder, ".wav", True))
                files.extend(list_all_files(downsample_folder, ".MP3", True))
                files.extend(list_all_files(downsample_folder, ".WAV", True))
            else:
                splits = ["vad", "vad_speech", "vad_balanced", "getrennt"]
                for element in splits:
                    downsample_folder = os.path.join(config["data_folder"], element)
                    if os.path.isdir(downsample_folder):
                        files = list_all_files(downsample_folder, ".mp3", True)
                        files.extend(list_all_files(downsample_folder, ".wav", True))
                        files.extend(list_all_files(downsample_folder, ".MP3", True))
                        files.extend(list_all_files(downsample_folder, ".WAV", True))

            stepsize = 300
            n_splits = len(files) / stepsize
            files_split = np.array_split(np.array(files), n_splits)
            for parts in files_split:
                tmp_files = list()
                output_files = list()

                for filename in parts:
                    tmp_files.append(torchaudio.load(filename))

                for (data, sr) in tmp_files:
                    data = torchaudio.transforms.Resample(sr, samplerate).forward(data)
                    output_files.append(data)

                for data, filename in zip(output_files, parts):
                    if filename.endswith("mp3"):
                        os.system("rm " + filename)
                        filename = filename.replace(".mp3", ".wav")
                    torchaudio.save(filename, data[0].unsqueeze(0), samplerate)

                del tmp_files
                del output_files
