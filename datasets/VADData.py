import os
import random
import shutil
import pandas as pd


class VADData():

    big_noise = [
        "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip",
        "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip",
        "https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip",
        "https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip",
        "https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip",
        "https://zenodo.org/record/2529934/files/FSDnoisy18k.meta.zip"]
    big_noise_folder = ["./noise_files/FSDKaggle/", "./noise_files/FSDKaggle/",
                        "./noise_files/FSDKaggle/", "./noise_files/FSDnoisy/",
                        "./noise_files/FSDnoisy/", "./noise_files/FSDnoisy/"]

    # big_speech = ["https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/de.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/fr.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/es.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/it.tar.gz"]
    big_speech = ["https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/it.tar.gz"]

    def __init__(self):
        self.big = False
        self.small = False
        self.make_structure()

    def downsample(self):
        print("Start with downsample of VAD")
        for path, subdirs, files in os.walk("./vad_data_balanced/"):
            for name in files:
                if name.endswith("wav") and not name.startswith("."):
                    os.system("ffmpeg -y -i " + os.path.join(path, name) +
                              " -ar 16000 -loglevel quiet " + os.path.join(path, "new" + name))
                    os.system("rm " + os.path.join(path, name))
                    os.system("mv " + os.path.join(path, "new" + name) + " " + os.path.join(path, name))
                elif name.endswith("mp3") and not name.startswith("."):
                    os.system("ffmpeg -y -i " + os.path.join(path, name) +
                              " -ar 16000 -ac 1 -loglevel quiet " + os.path.join(path, name.replace(".mp3", ".wav")))
                    os.system("rm " + os.path.join(path, name))
        print("Finished Downsample")

    def make_structure(self):
        os.system("mkdir -p ./noise_files")
        os.system("mkdir -p ./speech_files")
        os.system("mkdir -p ./vad_data_balanced/")

    def download(self, link, certificate=""):
        os.system("wget " + link + " " + certificate)

    def unzip(self, file, destination):
        os.system("unzip -P pass " + file + " -d " + destination)

    def untar(self, file, destination):
        os.system("tar -xvzf " + file + " -C " + destination)

    def get_small_dataset(self):
        self.small = True
        for i in range(1, 10):
            self.download("https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio." + str(i) + ".zip")
            pass
        self.unzip("'*.zip'", "./noise_files/")
        os.system("rm *.zip")

        self.download("https://zeos.ling.washington.edu/corpora/UWNU/uwnu-v2.tar.g", "--no-check-certificate")
        self.untar("uwnu-v2.tar.gz", "./speech_files/")

    def get_big_dataset(self):
        self.big = True
        for link, folder in zip(self.big_noise, self.big_noise_folder):
            self.download(link)
            self.unzip("*.zip", folder)
            os.system("rm *.zip")

        for link in self.big_speech:
            self.download(link)
            self.untar("*.tar.gz", "./speech_files/")
            os.system("rm *.tar.gz")
            os.system("mv ./speech_files/cv-corpus-5.1-2020-06-22/ ./speech_files/mozilla/")

        if not os.path.exists("./noise_files/FSD"):

            kagglecomp_train = pd.read_csv("./noise_files/FSDKaggle/FSDKaggle2018.meta/train_post_competition.csv")
            kagglecomp_test = pd.read_csv(
                "./noise_files/FSDKaggle/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv")
            kagglecomp_train_remove = list()
            kagglecomp_test_remove = list()

            for idx, element in kagglecomp_train.iterrows():
                if (element["label"] == "Laughter") or (element["label"] == "Applause") or (element["label"] == "Telephone") or\
                        (element["label"] == "Acoustic_guitar") or (element["label"] == "Fireworks") or \
                        (element["label"] == "Hi-hat") or (element["label"] == "Squeak") or (element["label"] == "Tearing") or \
                        (element["label"] == "Writing"):
                    kagglecomp_train_remove.append(idx)

            for idx, element in kagglecomp_test.iterrows():
                if (element["label"] == "Laughter") or (element["label"] == "Applause") or (element["label"] == "Telephone") or \
                        (element["label"] == "Acoustic_guitar") or (element["label"] == "Fireworks") or \
                        (element["label"] == "Hi-hat") or (element["label"] == "Squeak") or (element["label"] == "Tearing") or \
                        (element["label"] == "Writing"):
                    kagglecomp_test_remove.append(idx)

            kagglecomp_train = kagglecomp_train.drop(kagglecomp_train.index[kagglecomp_train_remove], 0)
            kagglecomp_test = kagglecomp_test.drop(kagglecomp_test.index[kagglecomp_test_remove], 0)

            kagglecomp_train_remove = list()

            for idx, element in enumerate(kagglecomp_train["freesound_id"]):
                if element in list(kagglecomp_test["freesound_id"]):
                    kagglecomp_train_remove.append(idx)

            # working with FSDNoise
            kagglenoise_train = pd.read_csv("./noise_files/FSDnoisy/FSDnoisy18k.meta/train.csv")
            kagglenoise_test = pd.read_csv("./noise_files/FSDnoisy/FSDnoisy18k.meta/test.csv")

            path = "./noise_files/FSD"
            os.system("mkdir " + path)

            for idx, element in kagglecomp_test.iterrows():
                os.system("cp " + os.path.join("./noise_files/FSDKaggle/FSDKaggle2018.audio_test",
                                               element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

            for idx, element in kagglecomp_train.iterrows():
                os.system("cp " + os.path.join("./noise_files/FSDKaggle/FSDKaggle2018.audio_train",
                                               element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

            for idx, element in kagglenoise_test.iterrows():
                os.system("cp " + os.path.join("./noise_files/FSDnoisy/FSDnoisy18k.audio_test",
                                               element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

            for idx, element in kagglenoise_train.iterrows():
                os.system("cp " + os.path.join("./noise_files/FSDnoisy/FSDnoisy18k.audio_train",
                                               element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

            os.system("rm -r ./noise_files/FSDKaggle")
            os.system("rm -r ./noise_files/FSDnoisy")

    def copy_to_destination(self, destination_dict):
        for key, value in destination_dict.items():
            data_dir = "./vad_data_balanced/" + key
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            for f in value:
                shutil.copy(f, data_dir)

    def load_Folder(self, dictionary):
        files = []
        for path, subdirs, files in os.walk(dictionary):
            for name in files:
                if (name.endswith("wav") or name.endswith("mp3")) and not name.startswith("."):
                    files.append(os.path.join(path, name))
        return files

    def split_dataset(self):

        # directories with original data
        noise_dir = "./noise_files/"
        speech_dir = "./speech_files/"

        # list all noise  and speech files
        noise_files = self.load_Folder(noise_dir)

        speech_files = self.load_Folder(speech_dir)

        # randomly shuffle noise and speech files and split them in train,
        # validation and test set
        random.shuffle(noise_files)
        random.shuffle(speech_files)

        nb_noise_files = len(noise_files)
        nb_train_noise = int(0.6 * nb_noise_files)
        nb_dev_noise = int(0.2 * nb_noise_files)

        train_noise = noise_files[:nb_train_noise]
        dev_noise = noise_files[nb_train_noise:nb_train_noise+nb_dev_noise]
        test_noise = noise_files[nb_train_noise+nb_dev_noise:]

        train_speech = speech_files[:nb_train_noise]
        dev_speech = speech_files[nb_train_noise:nb_train_noise+nb_dev_noise]
        test_speech = speech_files[nb_train_noise+nb_dev_noise:nb_noise_files]

        train_bg_noise = train_noise[:100]
        dev_bg_noise = dev_noise[:100]
        test_bg_noise = test_noise[:100]

        destination_dict = {"train/noise": train_noise,
                            "train/speech": train_speech,
                            "dev/noise": dev_noise,
                            "dev/speech": dev_speech,
                            "test/noise": test_noise,
                            "test/speech": test_speech,
                            "train/background_noise": train_bg_noise,
                            "dev/background_noise": dev_bg_noise,
                            "test/background_noise": test_bg_noise}

        for key, value in destination_dict.items():
            data_dir = "./vad_data_balanced/" + key
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            for f in value:
                shutil.copy(f, data_dir)
        # self.copy_to_destination(destination_dict)
