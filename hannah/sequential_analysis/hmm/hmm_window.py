#
# Copyright (c) 2023 Hannah contributor.
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
import numpy as np
import pandas as pd
from hmm_visualization import (
    plot_accuracies_cnn_hmm,
    visualize_confusion_matrix,
    visualize_error_ratio,
    visualize_single_study,
    plot_delay_distribution
)
from window_size_sweep import window_size_sweep
from sklearn.metrics import confusion_matrix, mean_absolute_error, recall_score, precision_score, accuracy_score, f1_score
from tabulate import tabulate
from viterbi import viterbi_window
import argparse
import sys

def main():

    ##############
    # Read Input #
    ##############
    parser = argparse.ArgumentParser(description="Performs Post-Processing of the CNN output on time-dependent data with a HMM and Viterbi decoding.")
    parser.add_argument("--cnn_output_dir", type=str, help="Path to the output of the trained CNN.")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_small_075")
    parser.add_argument("--window_size", type=int, default=300, help='Window size used during Viterbi decoding.')
    parser.add_argument("--window_size_sweep", type=bool, default=False, help='Whether to perform a window size sweep.')
    parser.add_argument("--class_of_interest", type=int, help='The class number for which the delay should be tracked, e.g. 1 for seizure detection, 3 for small intestine in RI dataset.')
    parser.add_argument("--quantization", type=bool, default=False, help="Whether to quantize inputs for the viterbi algorithm")
    parser.add_argument("--quant_type", type=str, default='rounding', help="Quantizing method. Options are: linear_scaling and random")
    args = parser.parse_args()
    CNN_DIR = args.cnn_output_dir
    CNN_NAME = args.model_name
    SIZE = args.window_size
    WINDOW_SWEEP = args.window_size_sweep
    CLASS_OF_INTEREST = args.class_of_interest
    QUANTIZATION = args.quantization
    QUANT_TYPE = args.quant_type

    path_train = CNN_DIR + CNN_NAME + '_cnn_train_output'
    path_test = CNN_DIR + CNN_NAME + '_cnn_test_output'

    # Processing of CSV Files - first header row = column names 
    df_train = pd.read_csv(path_train)
    df = pd.read_csv(path_test) 

    def remove_header_rows(df_temp):
        original_size = df_temp.shape[0]
        column_set = set(df_temp.columns)
        matching_rows = df_temp[df_temp.isin(column_set).any(axis=1)]  # Check if any element in the row is in the column names (order not considered here)

        matching_mask = (matching_rows == pd.Series(df_temp.columns, index=df_temp.columns)).all(axis=1) # mask for matching rows also in terms of the order of columns

        if not matching_mask.all(): # If there is any row, where the order is different - warning!

            print(Warning, 'Different order in rows. Rearranging DataFrame')
            partially_matching_rows = matching_rows[~matching_mask]

            for idx in partially_matching_rows.index: # Rearrange columns starting from the partially matching row until next header row or until end of df
                current_header_row_idx = matching_rows.index.get_loc(idx)
                if matching_rows.index[len(matching_rows.index)-1] == current_header_row_idx: # last header row is in wrong order - reordering until end of df
                    next_header_row_idx = len(df_temp.index) - 1 
                else:
                    next_header_row_idx = matching_rows.index[current_header_row_idx + 1]
                df_temp_partial = df_temp.iloc[idx:next_header_row_idx] # Get all rows in df until the next header row in csv file occurs
                df_temp_partial.columns = df_temp_partial.loc[idx]  # Set the first row as column names
                df_temp_partial = df_temp_partial.drop(idx)
                df_temp_partial = df_temp_partial[df_temp.columns] # Same order as main df
                df_temp.loc[df_temp_partial.index] = df_temp_partial # Replace in old df

        df_temp = df_temp.drop(matching_rows[matching_mask].index).reset_index(drop=True) # Drop all rows that exactly match
        df_temp = df_temp.drop(matching_rows[~matching_mask].index).reset_index(drop=True) # Drop all rows that exactly match
        # Rename column names in order to fit the names used in the following.
        df_temp.rename(columns={'study_id': 'id', 'preds_cnn': 'preds'}, inplace=True)
        df_temp['labels'] = pd.to_numeric(df_temp['labels'])
        df_temp['preds'] = pd.to_numeric(df_temp['preds'])

        assert df_temp.shape[0] + len(matching_rows) == original_size # Assure that nothing is missing!

        return df_temp

    df_train = remove_header_rows(df_train)
    df = remove_header_rows(df)


    ################
    # Preparations #
    ################
    # Training CNN Stats
    cm_train = get_cm(df_train.labels, df_train.preds)

    # Transitions, Emissions and Start Probabilities
    cm_train = get_cm(df_train.labels, df_train.preds)
    logA = get_log_transition_matrix(cm_train.shape)
    emissions = get_emission_matrix(cm_train)
    logB = np.log(emissions)
    start_v = np.ones(cm_train.shape[0])*(10**-10)
    start_v[0] = 1 # We always want to start at the first organ.
    logP = np.log(start_v)

    if QUANTIZATION:
        logP, logA, logB = quantize_matrices(logP, logA, logB, QUANT_TYPE)

    ####################
    # Viterbi Decoding #
    ####################
    study_id = df['id'].iloc[0]
    df_single_study = df.loc[df["id"] == study_id]
    visualize_single_study(
        df_single_study, logA, logB, logP
    )  # Predicted classes over time

    (
        true_labels,
        preds_all,
        false_predictions,
        accuracies_per_id,
        delays
    ) = compute_preds(  # viterbi
        df, logA, logB, logP, SIZE, CLASS_OF_INTEREST, 1
    )


    #################
    # Visualization #
    #################
    # Evaluation with CNN observations only
    cm_cnn = get_cm(y_true=df.labels, y_pred=df.preds)
    print_metrics(y_true=df.labels, y_pred=df.preds, delays=0, method='CNN', cm=cm_cnn)

    # Evaluation with additional post processing by HMM and Viterbi algorithm
    cm_hmm = get_cm(y_true=true_labels, y_pred=preds_all)
    print_metrics(y_true=true_labels, y_pred=preds_all, delays=np.median(np.absolute(delays)), method='CNN+HMM', cm=cm_hmm)

    cnn_err_id = cnn_acc_study(df)
    plot_accuracies_cnn_hmm(
        accuracies_per_id, cnn_err_id, df.id.unique(), "accuracies_cnn_hmm.pdf"
    )

    visualize_confusion_matrix(cm_hmm)
    visualize_error_ratio(false_predictions)  # how many false predictions?

    plot_delay_distribution(delays)
    
    if WINDOW_SWEEP:
        window_size_sweep(df=df, logP=logP, log_transitions=logA, emissions=emissions)


def quantize_matrices(logP, logA, logB, quant_method):
    
    quant_res = 4
    min_logP = np.min(logP)
    logP = np.array([quantize_value(i, min_logP, -1*(pow(2,quant_res) - 1), quant_method) for i in logP])
    min_AB = np.sort(np.unique(np.append(logA.flatten(), logB.flatten())))[1]
    logA = np.array([[quantize_value(i, min_AB, -1*(pow(2,quant_res) - 1), quant_method) for i in row] for row in logA])
    logB = np.array([[quantize_value(i, min_AB, -1*pow(2,quant_res), quant_method) for i in row] for row in logB])

    return logP, logA, logB


def quantize_value(value, max_val, max_quant, method="rounding"):
    #print(f"scaling {value} from 0-{max_val} up to 0-{max_quant}, then rounding")
    if max_val == 0:
        return 0
    if method == "rounding":
        return round(value)
    elif method == "linear_scaling":
        return round((value / max_val) * max_quant)


def get_cm(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return cm

def get_emission_matrix(cm):

    cm = np.where( cm==0 , 1, cm)  # To avoid division by zero with np.log and subsequent setting to -inf
    em = np.zeros(cm.shape)

    for i in range(cm.shape[0]):
        em[i] = cm[i] / sum(cm[i])

    assert np.mean(np.sum(em, axis=1)) == 1

    return em

def get_log_transition_matrix(shape):

    tm = np.zeros(shape)

    for i in range(tm.shape[0]-1): # The last row is special.
        tm[i,i] = np.log(0.999) # Values obtained from grid search.
        tm[i,i+1] = np.log(1 - 0.999)

    tm = np.where(tm==0., -(10**10), tm)
    tm[shape[0]-1, shape[1]-1] = np.log(1)

    return tm

def print_metrics(y_true, y_pred, delays, method, cm):

    # Compute metrics
    average = 'macro'
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average=average) 
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    mae = mean_absolute_error(y_true, y_pred)

    # Print metrics
    print(
        tabulate(
            [
                ["Accuracy", accuracy],
                ["Sensitivity", sensitivity],
                ["Precision", precision],
                ["F1-Score", f1],
                ["Mean Absolute Error", mae],
                ["Median Delay [# input samples]", delays],
            ],
            headers=[
                f"\n Results with {method}",
            ],
            tablefmt="outline",
        )
    )
    print('Confusion Matrix: \n', cm)


def cnn_acc_study(df):
    # Compute accuracies of CNN per study 
    studies = df.id.unique()
    err_rate_per_id = []

    for i in range(len(studies)):
        df_temp = df.loc[df["id"] == studies[i]]
        correct_preds_study = (
            df_temp.labels.to_numpy() == df_temp.preds.to_numpy()
        ).sum()
        err_rate_per_id.append(correct_preds_study / len(df_temp.labels))

    return err_rate_per_id


def compute_delay(y_true, y_pred, class_of_interest):

    assert len(y_true) == len(y_pred)

    onset_idx_label = np.where(y_true==class_of_interest)[0][0]
    onset_idx_pred = np.where(class_of_interest==y_pred)[0][0]
    delay = onset_idx_pred-onset_idx_label
    print(delay)

    return delay


def compute_preds(df, logA, logB, logP, size=300, class_of_interest=3, step=1):
    studies = df.id.dropna().unique()
    preds_all = []
    true_label_all = []
    false_predictions = []
    accuracies_per_id = []
    delays = []

    for i in range(len(studies)):
        print(" Study: ", studies[i])
        df_temp = df.loc[df["id"] == studies[i]]
        preds_viterbi = []
        preds_viterbi = viterbi_window(
                df_temp["preds"].to_numpy()[0::step],
                logP=logP,
                logA=logA,
                logB=logB,
                size=size,
                class_of_interest=class_of_interest
            )
        true_labels_study = df_temp.labels.to_numpy()[0::step]
        preds_all.extend(preds_viterbi)
        true_label_all.extend(true_labels_study)

        number_false_preds = (true_labels_study != preds_viterbi).sum()
        number_labels = len(df_temp.labels[0::step])
        false_predictions.append(
            (
                number_false_preds,
                number_labels,
            )
        )
        delays.append(compute_delay(true_labels_study, preds_viterbi, class_of_interest))
        accuracies_per_id.append(accuracy_score(true_labels_study, preds_viterbi))
        print('Accuracy: ', accuracy_score(true_labels_study, preds_viterbi))

    return true_label_all, preds_all, false_predictions, accuracies_per_id, delays


if __name__ == "__main__":
    main()
