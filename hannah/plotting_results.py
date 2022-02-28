import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import json


def barplot(data):
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=0.9)
    plt.rcParams['font.family'] = 'serif'
    fig = data.plot(x='Layer', y=['Huffman encoding', '32-Bit encoding'], kind='bar', width=0.8, color=['midnightblue', 'slategrey'])
    fig.bar_label(fig.containers[0], fontsize=7, padding=1)
    fig.bar_label(fig.containers[1], fontsize=7, padding=1)
    fig.set_xticklabels(rotation=45, labels=data['Layer'])
    fig.set_xlabel('Feature maps')
    fig.set_ylabel('Bits')
    fig.legend()
    fig.set_title('Huffman encoding of feature maps during forward pass', weight='bold', pad=15)
    fig.get_figure().savefig('/local/wernerju/hannah/hist_features_huff')
    print(tabulate(data, headers='keys', tablefmt='psql'))


def lineplot(data):
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=0.9)
    plt.rcParams['font.family'] = 'serif'
    res8 = data[['test_accuracy', 'model', 'cluster_amount']].loc[(data['max_epochs'] == 10) & (data['model'] == 'tc-res8')].sort_values(by=['cluster_amount'])
    # data = data.loc[(data['max_epochs'] == 10) & (data['model'] == 'tc-res8')].sort_values(by=['cluster_amount'])
    res20 = data[['test_accuracy', 'model', 'cluster_amount']].loc[(data['max_epochs'] == 10) & (data['model'] == 'tc-res20')].sort_values(by=['cluster_amount'])
    ax = res8.plot(x='cluster_amount', y='test_accuracy', style='.-', markersize=8, color='midnightblue', label='TC-ResNet8')
    fig = res20.plot(ax=ax, x='cluster_amount', y='test_accuracy', style='s-', markersize=4, color='slategrey', label='TC-ResNet20')
    fig.set_xlabel('Number of K-means cluster per layer')
    fig.set_ylabel('Test accuracy')
    fig.set_title('Accuracy of TC-ResNet8 for variying number of cluster with keyword spotting datset', weight='bold', pad=15)
    # fig.set_ylim(ymax=1)
    fig.get_figure().savefig('/local/wernerju/hannah/cluster_acc_tcres8.png')


def main():
    with open('/local/wernerju/hannah/test_results') as f:
        results = pd.DataFrame([json.loads(line) for line in f]).drop_duplicates()  # Load runs as df and remove duplicates
    lineplot(results)


main()
