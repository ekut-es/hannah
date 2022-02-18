import logging
import torch
from hydra.utils import to_absolute_path, instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from hannah.huffman import Huffman_decoding, Huffman_encoding
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

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


def features(module):
    module.eval()

    # Get feature maps with forward hook
    out = {}
    def conv_output(name):
        def hook(model, input, output):
            out[name] = output.detach()
        return hook
    names = []
    for name, mod in module.named_modules():
        if hasattr(mod, "weight") and mod.weight != None and 'downsample' not in name:
            names.append(name.replace('model.layers.', ''))
            mod.register_forward_hook(conv_output(name))
    input_rand = torch.randn(1, 500)
    res = module(input_rand)

    # Encode features with Huffman
    features = []
    bits_features_original = []
    for k,v in out.items():
        bits_features_original.append(len(v.numpy().flatten())*32)
        features.append(list(v.numpy().flatten()))

    encoding, tree = Huffman_encoding(features)
    bits_features_encoded = []
    for i in encoding:
        bits_features_encoded.append(len(i))
    percentages = []
    for k in range(len(bits_features_encoded)):
        percentages.append(str(round(((1-(bits_features_encoded[k]/bits_features_original[k]))*100), 2))+' %')
    return bits_features_encoded, bits_features_original, names, percentages


def main():
    config = {'name': 'test', 'checkpoints': ['/local/wernerju/hannah/trained_models/test/tc-res8/last.ckpt'], 'noise': [], 'output_dir': 'eval', 'default_target': 'hannah.modules.classifier.StreamClassifierModule'}
    seed_everything(1234, workers=True)
    checkpoint_path = '/local/wernerju/hannah/trained_models/test/tc-res8/last.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = checkpoint["hyper_parameters"]

    if "_target_" not in hparams:
        target = config.default_target
        logging.warning("Target class not given in checkpoint assuming: %s", target)
        hparams["_target_"] = target

    hparams["num_workers"] = 8
    module = instantiate(hparams, _recursive_=False)
    module.setup("test")
    module.load_state_dict(checkpoint["state_dict"])

    # until this point, code was taken from eval.py
    # -----
    bits_features_encoded, bits_features_original, names, percentages = features(module)
    #decoding = Huffman_decoding(encoding, tree)

    data = pd.DataFrame(
    {'Layer': names,
    'Huffman encoding': bits_features_encoded,
    '32-Bit encoding': bits_features_original,
    'Compression rate': percentages
    })
    barplot(data)
    #print(bits_features_encoded, bits_features_original)

main()