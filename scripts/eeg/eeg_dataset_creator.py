#
# Copyright (c) 2023 Hannah contributors.
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


import mne
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import os, glob, h5py, gc, logging, argparse
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter


def save_to_hdf5(split, filename, x, y, trainval_splits=False, seed=None):    
    """
    Saves the data x and labels y to the file `DATA_OUTPUT_DIR/{split}/{filename}.hdf5` under `split`. 
    If trainval_splits is True, splits the data into a 1:4 validation-training split.
    The subsets can be accessed while reading the file, using the following
    
    >>> file = h5py.File(path/to/file, "r")
    >>> train_x, train_y = file["train"]["x"], file["train"]["y"]
    >>> val_x, val_y = file["val"]["x"], file["val"]["y"]    
    """
    path = f"{DATA_OUTPUT_DIR}/{split}/{filename}.hdf5"
    savefile = h5py.File(path, "w")
    
    class_counts = np.unique(y, return_counts=True)[1]
    
    if not trainval_splits:
        savefile.create_dataset(f"{split}/x", data=x, dtype=np.float32)
        savefile.create_dataset(f"{split}/y", data=y, dtype=np.int8)
        logging.info(f"Saved file at {split}/{filename} (without splits): {list(x.shape)} {list(y.shape)}")
    else:
        tx, vx, ty, vy = train_test_split(x, y, train_size=0.8, stratify=y, random_state=seed)
        savefile.create_dataset(f"train/x", data=tx, dtype=np.float32)
        savefile.create_dataset(f"train/y", data=ty, dtype=np.int8)
        savefile.create_dataset(f"val/x", data=vx, dtype=np.float32)
        savefile.create_dataset(f"val/y", data=vy, dtype=np.int8)
        logging.info(f"Saved file at {split}/{filename} (with splits): {list(tx.shape)} {list(ty.shape)} {list(vx.shape)} {list(vy.shape)}")
       
    logging.info(f"{split}/{filename} net class counts: {class_counts}")
    os.system(f"echo \"{split}/{filename},{class_counts[0]},{class_counts[1]}\" >> {DATA_OUTPUT_DIR}/class_counts.csv")
    savefile.close()
    

class EEGFile:
    """
    Basic .edf file preprocessing class. 
    
    Takes the .edf file path as input, along with seizure limits, if the
    file contains one. Data is resampled according to `sampling_rate` (default None) and top 17 channels which 
    are common to all files are selected by setting `pick_common` to True (default True).
    
    Main functionality:
        1. collate_data(num_seconds):
            splits the data in the shape (17,X) into (N,17,256*num_seconds), where N = X//(256*num_seconds)
        2. get_labels(num_seconds):
            creates binary labels for each entry in (N,17,256*num_seconds), assigning 1 to any time instance where there is a seizure and 0 otherwise. Output shape (N,) 
    """
    def __init__(self, filepath, seizure_limits, use_windowing=False, sampling_rate=256, pick_common=True):
        data = mne.io.read_raw_edf(filepath, verbose=False, preload=True)
        if sampling_rate != 256: 
            data = data.resample(sampling_rate)
        else:
            assert data.info['sfreq'] == 256, "Check sampling rate"
            
        if pick_common:
            # Can raise ValueError for a few files in chb12
            data = data.pick([
                #"T7-P7","C3-P3",
                "FP1-F7","F7-T7","T7-P7","P7-O1",
                "FP1-F3","F3-C3","C3-P3","P3-O1",
                "FP2-F4","F4-C4","C4-P4","P4-O2",
                "FP2-F8","F8-T8","P8-O2",
                "FZ-CZ","CZ-PZ"
            ])
        if sampling_rate > 100:
            data = data.filter(l_freq=0.1, h_freq=50, h_trans_bandwidth=0.1, verbose=False)
        self.fname = filepath.split("/")[-1]

        self.data = data
        self.raw_data = data.get_data()

        self.channels = data.ch_names
        self.n_channels = len(self.channels)
        self.samp_rate = int(data.info['sfreq'])
        self.seizure_limits = seizure_limits
        self.seizure = True if self.seizure_limits[0] > 0 else False
        self.use_seizure_to_train = use_windowing
        
    def get_data(self, num_seconds=1, window_ictal_data=True, overlap=1):
        """
        splits the data in the shape (17,X) into (N,17,256*num_seconds), where N = X//(256*num_seconds)
        """
        s = int(num_seconds*self.samp_rate)
        data = np.split(self.raw_data, np.arange(s, self.raw_data.shape[1], s), axis=1)
        # data = np.stack(np.array_split(self.raw_data, len(self), axis=1))
        if self.raw_data.shape[1] % s != 0: data = data[:-1]
        data = np.stack(data)
        labels = self.get_labels(num_seconds=num_seconds) 
                
        if window_ictal_data and self.seizure and self.use_seizure_to_train:
            ictal_idx = np.where(labels == 1)[0]
            ictal_data_collected = np.hstack(data[ictal_idx])
            ictal_data = []
            seq_len = int(self.samp_rate*num_seconds)
            step_size = int(seq_len * (1-overlap)) if overlap < 1 else int(overlap)
                            
            for i in range(0, ictal_data_collected.shape[1]-seq_len+1, step_size):
                ictal_data.append(ictal_data_collected[:,i:i+seq_len])
            ictal_data = np.stack(ictal_data)
            ictal_labels = np.ones(len(ictal_data))
  
            nonictal_idx = np.where(labels==0)[0]
            nonictal_data = data[nonictal_idx]
            nonictal_labels = labels[nonictal_idx]
            
            data = np.vstack((nonictal_data, ictal_data))
            labels = np.concatenate((nonictal_labels, ictal_labels))
        
        return torch.from_numpy(data), torch.from_numpy(labels)

    def get_labels(self, num_seconds=1):
        """
        creates binary labels for each entry in (N,17,256*num_seconds), assigning 1 to any time instance where there is a seizure and 0 otherwise. Output shape (N,) 
        """
        labels = np.zeros(int(len(self)//num_seconds))
        if self.seizure: labels[int(self.seizure_limits[0]//num_seconds):int(self.seizure_limits[1]//num_seconds)] = 1
        return labels

    def __len__(self):
        "Returns length of raw data in seconds (data shape = n_channels x length)"
        return self.raw_data.shape[1] // self.samp_rate

    def __repr__(self):
        if self.seizure:
            return f"File: {self.fname}\nSampling rate: {self.samp_rate}\nSeizure in file: {self.seizure} ({self.seizure_limits})"
        else:
            return f"File: {self.fname}\nSampling rate: {self.samp_rate}\nSeizure in file: {self.seizure}"
        

if __name__ == "__main__":
    """
    Example usage: python eeg_dataset_creator.py --output_dir "./data/" --class_ratio 5 --data_length 1 
    """

    parser = argparse.ArgumentParser(description="Reads all .edf files of the chbmit dataset and saves three subsets--dev, retrain and test--in their respective directories. The `dev` subset is used for training, and contains a `full.hdf5` which contains the combined data in the directory. The `retrain` subset is used for patient-specific retraining, and contains data for each patient with a 4:1 training/validation set. The `dev` and `retrain` subsets contain the data in a fixed ratio determined by the `--class_ratio` parameter, which specifies the ratio of zeros to ones (labels). The `test` subset contains the remaining data (not in a fixed ratio).")
    parser.add_argument("--dataset_root_dir", type=str, default="/home/kohlibha/chbmit/", help="Path to the chmbit dataset")
    parser.add_argument("--output_dir", type=str, default="data/", help="Directory to save the output to")
    parser.add_argument("--data_length", type=float, default=1, help="Number of seconds per data point. For a sample rate of 256, the final data will be of the shape (D,C,256*data_length)")
    parser.add_argument("--window_ictal_data", action="store_true", help="When True, creates sliding windows on ictal regions to generate more samples")
    parser.add_argument("--overlap", type=float, default=1, help="If window_ictal_data is passed, this parameter controls the percentage of overlap when the sliding window is applied. If overlap is > 1, the value itself is passed as the window step size.") 
    parser.add_argument("--class_ratio", type=int, default=5, help="Ratio of zeros:ones in the final `dev` and `retrain` datasets.")
    parser.add_argument("--samp_rate", type=int, default=256, help="Sampling rate for the dataset. The data is already sampled at 256 Hz, use this argument only when the rate needed differs from this")
    parser.add_argument("--pick_all", action="store_true", help="When passed, picks all channels from all .edf files. When not passed, picks only common channels. This is required because some files do not have channels other than the common channels, and collecting their data along with the others will cause errors during their use. Only pass this argument when absolutely necessary")
    parser.add_argument("--collect_data_only", action="store_true", help="This option only executes the last portion of the script (data collection and creation of `full.hdf5`). Pass this argument if the dataset already exists in the output directory, and just collection needs to be done for the `dev` subset.")
    parser.add_argument("--num_patients", type=int, default=-1, help="Optional override to choose the number of patients to extract data from. Pass -1 to use all.")
    args = parser.parse_args()
        
    # Main arguments
    DATASET_ROOT_DIR = args.dataset_root_dir
    DATA_OUTPUT_DIR = args.output_dir
    DATA_LENGTH_SECONDS = args.data_length
    WINDOW_ICTAL = args.window_ictal_data
    OVERLAP = args.overlap
    CLASS_RATIO = args.class_ratio
    
    # Processing arguments  
    SAMP_RATE = args.samp_rate
    PICK_COMMON = not args.pick_all
    
    # Misc
    COLLECT_DATA_ONLY = args.collect_data_only
    NUM_PATIENTS = args.num_patients 
    if NUM_PATIENTS == -1:
        NUM_PATIENTS = None
    
    ## Creating the required directories
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_OUTPUT_DIR + "/dev", exist_ok=True)
    # os.makedirs(DATA_OUTPUT_DIR + "/val", exist_ok=True)
    os.makedirs(DATA_OUTPUT_DIR + "/retrain", exist_ok=True)
    os.makedirs(DATA_OUTPUT_DIR + "/test", exist_ok=True)

    # The following creates a logger which creates logs in the specified file. This is used
    # to keep track of which files are created where and the shapes of data stored in them
    logfile = f"{DATA_OUTPUT_DIR}/creation_logs.log"
    os.system(f"echo \"\" > {logfile}")
    logging.basicConfig(
        filename=logfile,
        encoding="utf-8",
        filemode="a+",
        level=logging.INFO,
        format="%(levelname)s(%(asctime)s):%(message)s",
        datefmt="%d/%m %I:%M:%S%p"
    )
    logging.info(f"Running script with the following options: \n{args}\n\n")
    print("Running script with the following options: ", args, sep='\n')
    print(f"Logfile at {logfile}")
    
    os.system(f"echo \"file,non-seizures,seizures\" > {DATA_OUTPUT_DIR}/class_counts.csv")
    
    if not COLLECT_DATA_ONLY:
        ## Main loop, runs only if COLLECT_DATA_ONLY is False
        patient_dirs = [DATASET_ROOT_DIR + p for p in os.listdir(DATASET_ROOT_DIR) if os.path.isdir(DATASET_ROOT_DIR + p)]
        # patient_dirs = ["/home/.../chb01/", "/home/.../chb02/", "/home/.../chb03/",...] (not in exact order)

        pbar = tqdm(patient_dirs[:NUM_PATIENTS])
        for n, patient_dir in enumerate(pbar):

            patient_id = patient_dir.split("/")[-1]
            path = patient_dir + f"/{patient_id}-summary.txt"

            pbar.set_description_str(f"Processing {patient_id}")        

            ## script 1: extract filepaths and seizure limits from summary
            with open(path,"r") as f:
                filedata = f.readlines()

            filedata = [x.replace("\n", "").strip().lower() for x in filedata]
            # All extra whitespaces removed, all lines made lowercase

            relevant_lines = []
            for line in filedata:
                if line.startswith("file name"):
                    relevant_lines.append(patient_dir + "/" + line.split(" ")[2])
                elif line.startswith("seizure"):
                    relevant_lines.append(int(line.split(" ")[-2]))
            # Extracting only relevant lines
            # relevant_lines now contains file names which are possibly followed by numbers
            # if a filename is followed by a number, then it contains a seizure whose limits
            # are denoted by the next two elements. If a filename is not followed by a number
            # it does not contain a seizure.
            # Example:
            # relevant_lines = ["chb01.edf","chb02.edf",1996,2036,...] 
            # in the above example, chb01.edf does not contain a seizure, chb02.edf contains 
            # a seizure from time 1996 to 2036.

            data = []
            for n,s in enumerate(relevant_lines):
                # Check for filename
                if isinstance(s,str):
                    if n == len(relevant_lines)-1:   
                        # last element of list is a filename ==> no seizure
                        data.append([s,0,0])            
                    else:
                        if isinstance(relevant_lines[n+1], int):  
                            # filename followed by integer ==> seizure limits in the next two lines
                            data.append([s,relevant_lines[n+1],relevant_lines[n+2]])
                        else:                                     
                            # filename not followed by integer ==> no seizure
                            data.append([s,0,0])  

            df = pd.DataFrame(data, columns=["filepath", "start", "end"])
            seizure_file_idx = df[df['start'] != 0].index
            num_seizures = len(seizure_file_idx)
            
            if WINDOW_ICTAL:
                if num_seizures <=2:
                    logging.warn(f"Patient {patient_id} skipped due to insuffient data for creating splits (Only {num_seizures} seizures found, 3 minimum required)")
                    continue 
                elif num_seizures == 3:
                    test_choices = [seizure_file_idx[-1]]
                    val_choices = [seizure_file_idx[-2]]
                elif num_seizures <=5:
                    test_choices = seizure_file_idx[-2:]
                    val_choices = [seizure_file_idx[-3]]
                else:
                    test_choices = seizure_file_idx[-2:]
                    val_choices = seizure_file_idx[-4:-2]                

                logging.info(f"Total {len(seizure_file_idx)} seizures found for patient {patient_id}. Using {len(test_choices)} for testing and {len(val_choices)} for validation.")
            
            ### script 2: reading, processing, saving
            filepaths, starts, ends = df.values.T
            data, labels = torch.tensor(()), torch.tensor(())
            testdata, testlabels = torch.tensor(()), torch.tensor(())
            retraindata, retrainlabels = torch.tensor(()), torch.tensor(())
        
            inner_pbar = tqdm(filepaths,leave=False)
            for (idx, filepath), start, end in zip(enumerate(inner_pbar), starts, ends):
                inner_pbar.set_description_str(filepath.split("/")[-1])
                if WINDOW_ICTAL:
                    use_train_val = (idx not in test_choices)
                    use_val = (idx in val_choices)
                else:
                    use_train_val = True
                    use_val = False
                
                try: 
                    # Channel mismatches can cause ValueError (occurs when processing chb12)
                    file = EEGFile(
                        filepath, 
                        seizure_limits=(start, end), 
                        use_windowing=use_train_val,
                        sampling_rate=SAMP_RATE, 
                        pick_common=PICK_COMMON
                    )
                    if not use_train_val and WINDOW_ICTAL:
                        logging.info(f"Seizure in file {file.fname} used for testing (sliding window not applied)")
                except ValueError as e: 
                    # Channel mismatches can cause ValueError (occurs when processing chb12)
                    logging.warning(f"File {filepath} skippped (Error: {e})")
                    continue
                
                file_data, file_labels = file.get_data(num_seconds=DATA_LENGTH_SECONDS, window_ictal_data=WINDOW_ICTAL, overlap=OVERLAP)
                
                if use_train_val:
                    if use_val:
                        retraindata = torch.cat((retraindata, file_data))
                        retrainlabels = torch.cat((retrainlabels, file_labels))
                    else:
                        data = torch.cat((data, file_data))
                        labels = torch.cat((labels, file_labels))
                else:
                    testdata = torch.cat((testdata, file_data))
                    testlabels = torch.cat((testlabels, file_labels))
                    
            # data and labels now contain all the data and labels from all files of a particular patient
            class_counts = np.unique(labels, return_counts=True)[1]
            ratio = class_counts[0] / class_counts[1]  
            
            CLASS_RATIO_ = int(np.floor(min(CLASS_RATIO, ratio)))
            
            # Shuffle data using the patient_id as a seed (reproducibility)
            torch.manual_seed(int(patient_id[-2:]))
            perm = torch.randperm(len(labels))
            data, labels = data[perm], labels[perm]

            # Get indices of ones and zeros, split into three subsets
            ones, zeros = torch.where(labels==1)[0], torch.where(labels==0)[0]
            
            if not WINDOW_ICTAL:
                # if windowing is set to false, the three splits need to be generated from one big tensor
                l1, l2 = int(0.6 * len(ones)), int(0.85 * len(ones))
                train_idx = torch.cat((ones[:l1],zeros[:CLASS_RATIO_*l1]))
                retrain_idx = torch.cat((ones[l1:l2],zeros[CLASS_RATIO_*l1:CLASS_RATIO_*l2]))
                extra_idx = torch.cat((ones[l2:], zeros[CLASS_RATIO_*l2:10*len(ones)]))
            
            # Save the three subsets
            
            if WINDOW_ICTAL:
                train_idx = torch.cat((ones, zeros[:int(CLASS_RATIO_*len(ones))]))
                save_to_hdf5("dev", patient_id, data[train_idx], labels[train_idx])#, trainval_splits=True, seed=int(patient_id[-2:]))
                save_to_hdf5("retrain", patient_id, retraindata, retrainlabels, trainval_splits=True, seed=int(patient_id[-2:]))
                save_to_hdf5("test", patient_id, testdata, testlabels)
            else:
                save_to_hdf5("dev", patient_id, data[train_idx], labels[train_idx])
                save_to_hdf5("retrain", patient_id, data[retrain_idx], labels[retrain_idx], trainval_splits=True, seed=int(patient_id[-2:]))
                save_to_hdf5("test", patient_id, data[extra_idx], labels[extra_idx])

            # Clear memory 
            del perm, ones, zeros, train_idx, data, labels
            gc.collect()

    ## creating a single compiled file with the data in dev, with train/val subsets
    for split in ["dev", "test"]:
        splitpath = DATA_OUTPUT_DIR + "/" + split 

        logging.info(f"Collecting data for all patients in {splitpath}")

        if os.path.exists(splitpath + "/full.hdf5"):
            os.remove(splitpath + "/full.hdf5")

        hdf_list = glob.glob(splitpath + "/*.hdf5")

        x_full = np.vstack([np.array(h5py.File(fn)[split]["x"]) for fn in hdf_list])
        y_full = np.vstack([np.array(h5py.File(fn)[split]["y"]).reshape(-1,1) for fn in hdf_list]).squeeze()

        save_to_hdf5(split, "full", x_full, y_full, trainval_splits=(split!="test"))