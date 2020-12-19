import os
import numpy as np
import torchaudio
from ..utils import list_all_files
from torchvision.datasets.utils import list_files, list_dir


class Downsample:
    def __init__(self):
        pass

    @classmethod
    def downsample(cls, config):
        if "downsample" not in config:
            return

        samplerate = config["downsample"]
        if samplerate > 0:
            print("downsample data begins")
            config["downsample"] = 0
            downsample_folder = ["train", "dev", "test"]
            torchaudio.set_audio_backend("sox")

            if len(config["data_split"]) != 0:
                downsample_folder = os.path.join(
                    config["data_folder"], config["data_split"]
                )
                files = list_all_files(downsample_folder, ".mp3", True)
                files.extend(list_all_files(downsample_folder, ".wav", True))
            else:
                splits = ["vad", "vad_speech", "vad_balanced", "getrennt"]
                for element in splits:
                    downsample_folder = os.path.join(config["data_folder"], element)
                    if os.path.isdir(downsample_folder):
                        files = list_all_files(downsample_folder, ".mp3", True)
                        files.extend(list_all_files(downsample_folder, ".wav", True))

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
                    torchaudio.save(filename, data[0], samplerate)

                del tmp_files
                del output_files