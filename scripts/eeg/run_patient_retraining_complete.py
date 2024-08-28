#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import glob, os, sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# sample script execution: python run_patient_retraining_full.py "tc-res20" "samp256"
# since custom expt name was not passed, the results will be stored @results/results_tc-res20_samp256_DDMM.csv and
# results/combined_confusion_matrix_tc-res20_samp256_DDMM.pdf (DDMM = todays day and month)


try:
    MODEL_NAME = sys.argv[1]  # Ex. tc-res20 / tc-res8 / ...
    DATASET_NAME = sys.argv[
        2
    ]  # Ex. samp256, samp256halfsec !! dataset used for training the main model stored in `trained_model/base_model/{MODEL_NAME}/`
    DATA_FOLDER = sys.argv[3]
    CKPT = sys.argv[4]  # Checkpoint file of pretrained model
except:
    raise EnvironmentError("Model name or dataset name missing")

try:
    EXPT_NAME = sys.argv[
        3
    ]  # custom experiment name as an identifier. Used while storing results generated in this file
except:
    EXPT_NAME = f"{datetime.now():%d%m}"


patient_list = os.listdir(f"{DATA_FOLDER}/chbmit/preprocessed/{DATASET_NAME}/retrain")
try:
    patient_list.remove(".ipynb_checkpoints")
except ValueError:
    pass
patient_list = [i[:-5] for i in patient_list]

for patient in patient_list[:10]:
    cmd = f"hannah-train dataset=chbmitrt dataset.data_folder={DATA_FOLDER} \
    features=identity trainer.max_epochs=20 model={MODEL_NAME} module.batch_size=8 optimizer=adamw \
    input_file={CKPT} experiment_id={patient} +dataset.retrain_patient={patient} \
    +module.test_batch_size=8 dataset.dataset_name={DATASET_NAME}"
    ret = os.system(cmd)
    if ret == 2:  # Ctrl+C pressed in terminal
        sys.exit()

# Compile csv with all loss metrics
pth = f"trained_models/{patient_list[0]}/{MODEL_NAME}/test_results.json"
file = json.load(open(pth, "r"))
for k in file.keys():
    file[k] = list(file[k].values())
file["patient"] = [patient_list[0]]

for patient in patient_list[1:]:
    try:
        pth = f"trained_models/{patient}/{MODEL_NAME}/test_results.json"
        f = json.load(open(pth, "r"))
        for k in f.keys():
            file[k].append(f[k]["0"])
        file["patient"].append(patient)
    except FileNotFoundError:
        print(patient, "skipped")

df = pd.DataFrame(file).set_index("patient")
print(df)
df.to_csv(f"results/results_{MODEL_NAME}_{DATASET_NAME}_{EXPT_NAME}.csv")

# Plot all confusion matrices
fig, ax = plt.subplots(5, 5, figsize=(15, 15))
ax = ax.flatten()

axes_off = np.vectorize(lambda ax: ax.axis("off"))
axes_off(ax)

for n, a in enumerate(ax[: len(df.index)]):
    pth = f"trained_models/{df.index[n]}/{MODEL_NAME}/test_confusion.png"
    conf_mat = plt.imread(pth)
    a.imshow(conf_mat)
    a.set_title(patient_list[n])

plt.savefig(
    f"results/combined_confusion_matrix_{MODEL_NAME}_{DATASET_NAME}_{EXPT_NAME}.pdf"
)
