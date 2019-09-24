
import os
import random
import shutil
import numpy as np

# directories with original data
noise_dir = "./noise_files"
speech_dir = "./speech_commands_v0.02"

# list all noise  and speech files
noise_files = []
for path, subdirs, files in os.walk(noise_dir):
    for name in files:
        if name.endswith("wav") and not name.startswith("."):
           noise_files.append(os.path.join(path, name))
 
speech_files = []
for path, subdirs, files in os.walk(speech_dir):
    if  "noise" not in subdirs:
        for name in files:
            if name.endswith("wav") and not name.startswith("."):
               speech_files.append(os.path.join(path, name))


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
    data_dir = "./vad_speech/" + key
    if not os.path.exists(data_dir):
       os.makedirs(data_dir)
    for f in value:
        shutil.copy(f, data_dir)
