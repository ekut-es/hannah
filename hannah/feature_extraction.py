import logging
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from hannah.huffman import Huffman_decoding, Huffman_encoding
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from torch import nn
from pytorch_lightning import Trainer
from hannah.models.factory.qat import ConvBnReLU1d, Conv1d


def barplot(data):
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=0.9)
    plt.rcParams['font.family'] = 'serif'
    fig = data.plot(x='Layer', y=['Huffman encoding', '32-Bit encoding'], kind='bar', width=0.8, color=['midnightblue', 'slategrey'])
    # fig.bar_label(fig.containers[0], fontsize=7, padding=1)
    # fig.bar_label(fig.containers[1], fontsize=7, padding=1)
    fig.set_xticklabels(rotation=45, labels=data['Layer'])
    fig.set_xlabel('Feature maps')
    fig.set_ylabel('Bits')
    fig.legend()
    fig.set_title('Huffman encoding of feature maps during forward pass', weight='bold', pad=15)
    fig.get_figure().savefig('/local/wernerju/hannah/hist_features_huff')
    print(tabulate(data, headers='keys', tablefmt='psql'))


def features(module):
    # -- Register Forward hook on ReLU layer to get their output --
    result = []

    def conv_output(name):
        def hook(model, input, output):
            test = dict()
            test[name] = output.detach()  # one dict = one layer output
            result.append(test)
        return hook

    for name, mod in module.named_modules():
        print(mod)
        if isinstance(mod, nn.ReLU):
            mod.register_forward_hook(conv_output(name))
    #breakpoint()
    # Input of test set
    trainer = Trainer(gpus=1, deterministic=True)
    reset_seed()
    trainer.test(model=module, ckpt_path=None)

    # -- Output separated into 3 lists, one for each ReLU layer --
    features = [None] * 3  # feature output separated into 3 lists, one for each layer
    bits_features_original = [None] * 3  # 32 bit encoding
    layer1 = []
    layer2 = []
    layer3 = []
    for value in result:
        for k, v in value.items():
            if k == 'model.convolutions.1.0.act':
                layer1.extend(list(v.cpu().numpy().flatten()))
                features[0] = layer1
                bits_features_original[0] = len(layer1)*32
            elif k == 'model.convolutions.2.0.act':
                layer2.extend(list(v.cpu().numpy().flatten()))
                features[1] = layer2
                bits_features_original[1] = len(layer2)*32
            elif k == 'model.convolutions.3.0.act':
                layer3.extend(list(v.cpu().numpy().flatten()))
                features[2] = layer3
                bits_features_original[2] = len(layer3)*32
    print('size', len(features))  # list of size 3

    encoding, tree, frq = Huffman_encoding(features)

    # -- Compute number of bits for encoded values --
    bits_features_encoded = []  # array of length 3, huffman encoded bits
    for i in encoding:
        bits_features_encoded.append(len(i))

    percentages = []
    for k in range(len(bits_features_encoded)):
        percentages.append(str(round(((1-(bits_features_encoded[k]/bits_features_original[k]))*100), 2))+' %')
    print('Size of Huffman Dictionary + encoded bits: ', (len(frq)*32)+(sum(bits_features_encoded)*2))
    print('Total number of original bits: ', sum(bits_features_original))
    print('Estimated compression in total: ', ((len(frq)*32)+(sum(bits_features_encoded)*2))/(sum(bits_features_original)))

    data = pd.DataFrame({
        'Layer': ['ReLU 1', 'ReLU 2', 'ReLU 3'],
        'Huffman encoding': bits_features_encoded,
        '32-Bit encoding': bits_features_original,
        'Compression rate': percentages
    })
    return data, tree, encoding, features


def main():
    # -------
    config = {'name': 'test', 'checkpoints': ['/local/wernerju/hannah/trained_models/test/conv_net_trax/best.ckpt'], 'noise': [], 'output_dir': 'eval', 'default_target': 'hannah.modules.classifier.StreamClassifierModule'}
    seed_everything(1234, workers=True)
    checkpoint_path = '/local/wernerju/hannah/trained_models/test/conv_net_trax/best.ckpt'
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

    # until this point, code was copied from eval.py
    # -----
    data, tree, encoding, feature = features(module)
    decoding = Huffman_decoding(encoding, tree)
    norm = 0
    for i in range(len(decoding)):
        norm += np.linalg.norm(np.asarray(decoding[i])-np.asarray(feature[i]))
    print('Difference of original and decoded values: ', norm)
    barplot(data)


main()
