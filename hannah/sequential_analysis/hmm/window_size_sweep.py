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
import numpy as np
import pandas as pd
from hmm_visualization import plot_window_distribution, plot_window_sweep
from viterbi import viterbi_window


def window_size_sweep(df, logP, log_transitions, emissions):
    mean_acc = list()
    mean_delays = list()
    all_delays = []
    min_max_accs = []
    sizes = [10, 50, 100, 200, 300] # Window sizes

    for size in sizes:
        (
            true_labels,
            preds,
            acc,
            m_delay,
            delay,
            min_max_acc,
        ) = compute_preds(  # viterbi
            df, logP, log_transitions, emissions, size=size
        )
        mean_acc.append(acc)
        mean_delays.append(m_delay)
        all_delays.append(delay)
        min_max_accs.append(min_max_acc)

    plot_window_sweep(mean_acc, mean_delays, sizes)
    plot_window_distribution(sizes, all_delays)


# plot number of false predictions per study
def compute_preds(df, logP, log_transitions, emissions, step=1, size=500):
    correct_preds = 0
    studies = df.id.unique()
    preds = []
    true_label = []
    false_predictions = []
    accuracies_per_id = []
    delays = []
    for i in range(len(studies)):
        print(" Study: ", studies[i])
        df_temp = df.loc[df["id"] == studies[i]]

        preds_viterbi, delay = viterbi_window(
            df_temp["preds"].to_numpy()[0::step],
            logP=logP,
            logA=log_transitions,
            logB=emissions,
            size=size,
        )

        delays.append(delay)
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
    print("Delays: ", np.mean(delays))

    return true_label, preds, acc, np.mean(delays), delays, accuracies_per_id
