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
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from hmm_visualization import plot_accuracies_grid_search
from hmm_window import get_emission_matrix
from viterbi import viterbi



def main():

    ##############
    # Read Input #
    ##############
    parser = argparse.ArgumentParser(description="Performs a grid search for the transition and emission probabilities on the CNN evaluations of the train set with Viterbi decoding without a window.")
    parser.add_argument("--cnn_output_dir", type=str, help="Path to the output of the trained CNN; e.g. /hannah/experiments/rhode_island/trained_models/EXPERIMENT_ID/")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_small_075")
    parser.add_argument("--values", type=float, default=[0.9, 0.95, 0.99, 0.999], nargs='+', help="Provide a list of possible non-logarithmic values to choose from during the grid search. A Permutation of all combinations is performed.")
    args = parser.parse_args()

    CNN_DIR = args.cnn_output_dir
    CNN_NAME = args.model_name
    VALUES = args.values
    path_train = CNN_DIR + CNN_NAME + '_cnn_train_output'

    df_train = pd.read_csv(path_train, names=["id", "preds", "labels"])
    cm_train = confusion_matrix(y_true=df_train.labels, y_pred=df_train.preds)
    cm_shape = cm_train.shape


    ################
    # Preparations #
    ################
    # Get emissions, logP and transitions
    emissions = get_emission_matrix(cm_train)

    start_v = np.ones(cm_shape[0])*(10**-10)
    start_v[0] = 1 # We always want to start at the first organ.
    logP = np.log(start_v)

    log_transitions = np.zeros(cm_shape)


    ###############
    # Grid Search #
    ###############
    accuracies = list()
    permutations = list()

    for i in itertools.product(VALUES, repeat=cm_shape[0]-1):
        for j in range(log_transitions.shape[0]-1): # The last row is special.
            log_transitions[j,j] = np.log(i[j]) 
            log_transitions[j,j+1] = np.log(1 - i[j])

        log_transitions = np.where(log_transitions==0., -(10**10), log_transitions) # for logarithmic matrix
        log_transitions[cm_shape[0]-1, cm_shape[1]-1] = np.log(1) # If the last organ is reached, we have to stay in this organ.

        true_label, preds, false_predictions, acc, accuracies_per_id = compute_preds(
            df_train, log_transitions, emissions, logP
        )
        accuracies.append(acc)
        permutations.append(str(i))  # all combinations

    plot_accuracies_grid_search(accuracies, permutations, "acc_grid_search.pdf")



def compute_preds(df, log_transitions, emissions, logP, step=1):
    correct_preds = 0
    studies = df.id.unique()
    preds = []
    true_label = []
    false_predictions = []
    accuracies_per_id = []
    for i in range(len(studies)):
        print(" Study: ", studies[i])
        df_temp = df.loc[df["id"] == studies[i]]
        preds_viterbi = viterbi(
            df_temp["preds"].to_numpy()[0::step],
            logP=logP,
            logA=log_transitions,
            logB=np.log(emissions),
        )
        correct_preds_study = (
            df_temp.labels.to_numpy()[0::step] == preds_viterbi
        ).sum()
        correct_preds += correct_preds_study
        preds.extend(preds_viterbi)
        true_label.extend(df_temp.labels.to_numpy()[0::step])
        false_predictions.append(
            (
                (df_temp.labels.to_numpy()[0::step] != preds_viterbi).sum(),
                len(df_temp.labels[0::step]),
            )
        )
        accuracies_per_id.append(correct_preds_study / len(df_temp.labels[0::step]))
    acc = correct_preds / len(df.labels[0::step])
    print("Accuracy:", acc)
    return true_label, preds, false_predictions, acc, accuracies_per_id


if __name__ == "__main__":
    main()
