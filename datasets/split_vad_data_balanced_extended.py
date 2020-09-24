
import os
import random
import shutil
import numpy as np
import pandas as pd
import wave
import contextlib

# directories with original data
noise_dir = "./noise_files"
speech_dir = "./speech_files"

#working with FSDKaggle

if not os.path.exists("./noise_files/FSD"):

    kagglecomp_train = pd.read_csv("./noise_files/FSDKaggle/FSDKaggle2018.meta/train_post_competition.csv")
    kagglecomp_test = pd.read_csv("./noise_files/FSDKaggle/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv")
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

    #working with FSDNoise

    kagglenoise_train = pd.read_csv("./FSDnoisy/FSDnoisy18k.meta/train.csv")
    kagglenoise_test = pd.read_csv("./FSDnoisy/FSDnoisy18k.meta/test.csv")

    path = "./noise_files/FSD"
    os.system("mkdir " + path)

    for idx, element in kagglecomp_test.iterrows():
        os.system("cp " + os.path.join("./noise_files/FSDKaggle/FSDKaggle2018.audio_test", element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

    for idx, element in kagglecomp_train.iterrows():
        os.system("cp " + os.path.join("./noise_files/FSDKaggle/FSDKaggle2018.audio_train",
                                       element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

    for idx, element in kagglenoise_test.iterrows():
        os.system("cp " + os.path.join("./FSDnoisy/FSDnoisy18k.audio_test", element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

    for idx, element in kagglenoise_train.iterrows():
        os.system("cp " + os.path.join("./FSDnoisy/FSDnoisy18k.audio_train",
                                       element["fname"]) + " " + os.path.join("./noise_files/FSD", element["fname"]))

    os.system("rm -rf ./noise_files/FSDKaggle")
    os.system("rm -rf ./FSDnoisy")

speech_files = []
for path, subdirs, files in os.walk(speech_dir):
    for name in files:
        if (name.endswith("wav") or name.endswith("mp3")) and not name.startswith("."):
           speech_files.append(os.path.join(path, name))

noise_files = []
for path, subdirs, files in os.walk(noise_dir):
    for name in files:
        if (name.endswith("wav") or name.endswith("mp3")) and not name.startswith("."):
           noise_files.append(os.path.join(path, name))



# randomly shuffle noise and speech files and split them in train,
# validation and test set
random.shuffle(noise_files)
random.shuffle(speech_files)

nb_noise_files = len(noise_files)
nb_train_noise = int(0.6 * nb_noise_files)
nb_dev_noise = int(0.2 * nb_noise_files)


#nb_speech_files = len(speech_files)
#nb_train_speech = int(0.6 * nb_speech_files)
#nb_dev_speech = int(0.2 * nb_speech_files)

train_noise = noise_files[:nb_train_noise]
dev_noise = noise_files[nb_train_noise:nb_train_noise+nb_dev_noise]
test_noise = noise_files[nb_train_noise+nb_dev_noise:]

#train_speech = speech_files[:nb_train_speech]
#dev_speech = speech_files[nb_train_speech:nb_train_speech+nb_dev_speech]
#test_speech = speech_files[nb_train_speech+nb_dev_speech:]

train_speech =speech_files[:nb_train_noise]
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

#for path, subdirs, files in os.walk("./vad_data_balanced/"):
#    for name in files:
#        if name.endswith("mp3") and not name.startswith("."):
#            os.system("sox -S -r 16000 " + os.path.join(path, name) + " " + os.path.join(path, name.replace(".mp3", ".wav")))
#            os.system("rm  " + os.path.join(path, name))
