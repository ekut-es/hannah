import os
import sys
import shutil
from ..utils import list_all_files, extract_from_download_cache


class NoiseDataset:
    def __init__(self):
        pass

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
                # remove old data because they could be incomplete
                if os.path.isdir(tut_target):
                    shutil.rmtree(tut_target)

                target_cache = os.path.join(downloadfolder_tmp, "TUT")

                for i in range(1, 10):
                    filename = (
                        "TUT-acoustic-scenes-2017-development.audio." + str(i) + ".zip"
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
