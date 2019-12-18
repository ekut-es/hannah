
import os
import random
import shutil
import numpy as np

# directories with original data
noise_dir = "./noise_files"
speech_dir = "./speech_files"

# list all noise  and speech files
noise_files = []
for path, subdirs, files in os.walk(noise_dir):
    for name in files:
        if name.endswith("wav"):
           noise_files.append(os.path.join(path, name))

speech_files = []
for path, subdirs, files in os.walk(speech_dir):
    for name in files:
        if name.endswith("wav"):
           speech_files.append(os.path.join(path, name))


# randomly shuffle the noise and speech files and split them in train,
# validation and test set
random.shuffle(noise_files)
random.shuffle(speech_files)

nb_noise_files = len(noise_files)
nb_train_noise = int(0.6 * nb_noise_files)
nb_dev_noise = int(0.2 * nb_noise_files)


nb_speech_files = len(speech_files)
nb_train_speech = int(0.6 * nb_speech_files)
nb_dev_speech = int(0.2 * nb_speech_files)

train_noise = noise_files[:nb_train_noise]
dev_noise = noise_files[nb_train_noise:nb_train_noise+nb_dev_noise]
test_noise = noise_files[nb_train_noise+nb_dev_noise:]

train_speech = speech_files[:nb_train_speech]
dev_speech = speech_files[nb_train_speech:nb_train_speech+nb_dev_speech]
test_speech = speech_files[nb_train_speech+nb_dev_speech:]


destination_dict = {"train/noise": train_noise,
                    "train/speech": train_speech,
                    "dev/noise": dev_noise,
                    "dev/speech": dev_speech,
                    "test/noise": test_noise,
                    "test/speech": test_speech}

for key, value in destination_dict.items():
    data_dir = "./vad_data/" + key
    if not os.path.exists(data_dir):
       os.makedirs(data_dir)
    for f in value:
        shutil.move(f, data_dir)