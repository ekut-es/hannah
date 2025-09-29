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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from viterbi import viterbi_window
import os

DIR = os.path.dirname(__file__)
file_path = os.path.join(DIR + '/figures_hmm/')
if not os.path.exists(file_path):
    os.makedirs(file_path)

def plot_accuracies_cnn_hmm(acc_hmm, acc_cnn, x, name="none.png"):

    if x.dtype != np.str_:
        x = x.astype(str)
    id_sort = np.argsort(acc_cnn)
    
    sns.set_theme()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    fig = plt.figure(figsize=(13, 8))

    plt.plot(
        x[id_sort],
        np.array(acc_hmm)[id_sort],
        linestyle="--",
        marker="o",
        label="CNN+HMM",
        color="#191970",
    )
    plt.plot(
        x[id_sort],
        np.array(acc_cnn)[id_sort],
        linestyle="--",
        marker="o",
        label="CNN",
        color="#027148",
    )
    plt.title("Accuracies CNN and HMM per study", fontweight="bold", fontsize=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel(ylabel="Accuracy", fontsize=15)
    plt.xlabel(xlabel="Patient studies", fontsize=15)
    plt.yticks(fontsize=15)
    #plt.ylim(bottom=0.5)
    plt.legend()
    plt.tight_layout()
    fig.savefig(file_path + name, format="pdf")


def plot_accuracies_grid_search(acc, x, name="none.png", ylim=1):
    sns.set_theme()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    fig = plt.figure(figsize=(30, 15))
    plt.plot(x, acc, linestyle="--", marker="o")
    plt.title("Accuracies of grid search", fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    #plt.ylim(top=ylim)
    plt.tight_layout()
    fig.savefig(file_path + name, format="pdf")


def visualize_single_study(df_single_study, log_A, logB, logP):
    preds_viterbi = viterbi_window(
        df_single_study["preds"].to_numpy(),
        logP=logP,
        logA=log_A,
        logB=logB,
        size=100,
        class_of_interest=2
    )
    df_single_study["preds_HMM"] = preds_viterbi
    sns.set_theme()
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout(pad=3.0)
    df_copy = (
        df_single_study.copy().reset_index()
    )  # use number of images starting from 0 for x axis
    ticks = np.arange(len(logP))
    df_copy.labels.plot(ax=axes[0], yticks=ticks, ylabel="Class")
    df_copy.preds.plot(ax=axes[1], yticks=ticks, ylabel="Class")
    df_copy.preds_HMM.plot(ax=axes[2], yticks=ticks, ylabel="Class")
    axes[0].set_title("True labels", fontweight="bold")
    axes[1].set_title("Predicted Labels from CNN", fontweight="bold")
    axes[2].set_title("Predicted Labels from CNN + HMM", fontweight="bold")
    fig.text(0.5, 0.04, "Number of input labels", ha="center")
    fig.savefig(file_path + "output_all.pdf", format="pdf")


def visualize_error_ratio(false_predictions):
    sns.set_theme()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    percentages_false_preds = []
    [percentages_false_preds.append((i / j) * 100) for i, j in false_predictions]
    false_preds_sorted = np.sort(np.array(percentages_false_preds))
    error_ratio = [
        (0.01 > false_preds_sorted).sum(),
        ((0.01 <= false_preds_sorted) & (false_preds_sorted < 0.1)).sum(),
        ((0.1 <= false_preds_sorted) & (false_preds_sorted < 1)).sum(),
        ((1 <= false_preds_sorted) & (false_preds_sorted < 10)).sum(),
        ((false_preds_sorted) >= 10).sum(),
    ]
    fig = plt.figure(figsize=(6, 4))
    plt.xticks(fontsize=8)
    ax = plt.bar(
        x=[
            "err<0.01%",
            "0.01%≤err<0.1%",
            "0.1%≤err<1%",
            "1%≤err<10%",
            "10%≤err",
        ],
        height=error_ratio,
        color="#274472",
        width=0.2,
    )
    plt.ylabel("Number of patient studies")
    plt.tight_layout()
    plt.title("Percentages of false predictions for all test studies", fontweight="bold")
    fig.savefig(file_path + "error_amount.pdf", bbox_inches="tight", format="pdf")


def visualize_confusion_matrix(confusion_m):
    fig = plt.figure()
    sns.heatmap(confusion_m, annot=True, cmap="Blues", fmt="g")
    fig.savefig(file_path + "confusion_matrix.pdf", format="pdf")


def plot_window_distribution(sizes, delays):
    sns.set_theme()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    fig, ax = plt.subplots()
    plt.boxplot(delays)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Delay [# samples]")
    ax.set_xlabel("Window size")
    plt.title("Distribution of delays across studies for different window sizes")
    plt.tight_layout()
    fig.savefig(file_path + "boxplot_window_sweep.pdf", format="pdf")


def plot_window_sweep(acc, delays, sizes):
    sns.set_theme()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(sizes, acc, color="#191970", linestyle="-", marker="o")
    ax2.plot(sizes, delays, color="#027148", marker="x")
    ax1.set_xlabel("Window size")
    ax1.set_ylabel("Average accuracy", color="#191970")
    ax2.set_ylabel("Average delay [# samples]", color="#027148")
    plt.title("Average accuracies and delays for different window sizes")
    plt.tight_layout()
    fig.savefig(file_path + "window_size_sweep.pdf", format="pdf")


def plot_delay_distribution(delays):
    sns.set_theme()
    fig = plt.figure()
    # Compute median delay - since it can be negative, use absolute values.
    median_delay = np.median(np.absolute(delays))

    sns.histplot(delays, bins=150)
    plt.xlabel('#Frames Delay')
    plt.ylabel('#VCE Studies') 
    plt.text(0.96, 0.85, f"Median: {median_delay}", fontsize=9,fontweight="bold", transform=plt.gca().transAxes, ha='right', va='top')
    plt.tight_layout()

    fig.savefig(file_path + "delay_hist.pdf", bbox_inches="tight", format="pdf")


def plot_likelihood_acc_delay(acc, delays, likelihood):
    sns.set_theme()
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(likelihood, acc, s=5, linewidths=4, c=delays, cmap=plt.cm.winter)
    plt.colorbar().set_label('Delays')
    plt.xlabel('Likelihood')
    plt.ylabel('Accuracy [%]')
    plt.title('Correlation: Likelihood - Accuracy - Delay' ,fontweight='bold')
    fig.savefig(file_path + "likelihood_acc_delay.pdf", format="pdf")