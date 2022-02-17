from cmath import nan
import numpy as np
from onnx import numpy_helper
import copy
import torch
from huffman import Huffman_encoding, Huffman_decoding
import argparse

def load_parameters(file_path):
    values = []
    lengths = []
    shapes = []
    original_weights = []
    clustered_model = torch.load(file_path, map_location='cpu')  # 'cpu' must be specified, otherwise a CUDA error can occur.
    for key, value in clustered_model["state_dict"].items():
        if "weight" in key and "downsample" not in key:
            shapes.append(value.shape)
            original_weights.append(value)
            lengths.append(len(value.numpy().flatten()))
            values.append(value.numpy().flatten())
    return values, lengths, shapes, original_weights


def replace_cluster_by_indices(parameters, cluster):
    ws = copy.deepcopy(parameters)
    ws_indexed = []
    index_LUT = np.full(shape=(len(ws), cluster+1), fill_value=0, dtype=float)  # needs to be float, otherwise, inserted values are automatically rounded
    for k in range(len(ws)):
        centers = np.unique(ws[k], return_counts=False)  # get unique cluster centers
        for j in range(len(centers)):  # fill LUT
            index_LUT[k,j] = centers[j]
        intermediate_layer = np.select([ws[k]==centers[x] for x in range(len(centers))], 
                    [i+1 for i in range(len(centers))], 
                    ws[k])  # replace with index 1 to i for i cluster
        ws_indexed.append(intermediate_layer)
    #print('LUT: ', index_LUT)
    return ws_indexed, index_LUT


def replace_indices_by_clusters(ws_ind, index_LUT, ws):
    ws_indexed = copy.deepcopy(ws_ind)
    ws_cluster = []
    for i in range(len(ws_indexed)):
        ws_indexed[i] = np.asarray(ws_indexed[i], dtype=float)
        intermediate_layer = (np.select([ws_indexed[i]==x for x in range(1, index_LUT.shape[1]+1)],
        [index_LUT[i,k] for k in range(index_LUT.shape[1])], ws_indexed[i]))
        ws_cluster.append(intermediate_layer)
    norm_cluster_original = 0
    for i in range(len(ws_cluster)):
        norm_cluster_original += np.linalg.norm(ws_cluster[i]-ws[i])
    print('Norm of original and decoded/clustered weights: ', norm_cluster_original)
    return ws_cluster


def rebuild_struc(decoding, index_LUT, ws, shapes, original_ws):
    ws_cluster = replace_indices_by_clusters(decoding, index_LUT, ws)
    # Rebuilding original structure
    ws_tensor = [0]*len(ws_cluster)
    norm = 0
    for j in range(len(ws_tensor)):
        x = np.asarray(ws_cluster[j], dtype=np.float32).reshape(shapes[j])
        ws_tensor[j] = torch.from_numpy(x)
        norm += np.linalg.norm(original_ws[j]-ws_tensor[j])
        #print(id(ws_cluster[j]), id(result[j]), id(original_ws[j])) # different objects
    print('Norm of decoded and restructured weights and original input weights: ', norm)
    return ws_cluster, ws_tensor
    

def calc_diff(hs):
    total_bits = 0
    for i in range(len(hs)):
        total_bits += len(hs[i])
    return total_bits



def main():
    # Only works for non-quantized models. For quantized models, please use backend encoding.
    parser = argparse.ArgumentParser(
        description="Replace Kmeans centroids by indices and perform Huffman encoding.")
    parser.add_argument("-i", "--filepath", dest="filename", type=str, required=True,
                    help="File path to state dict of trained clustered (non-quantized) model")
    parser.add_argument("-cluster", "--number_clusters", dest="n_clusters", type=int, required=True,
                    help="Number of k-means clusters that were used during training.")
    args = parser.parse_args()
    #/home/wernerju/.cache/pypoetry/virtualenvs/hannah-Wne_DMqI-py3.9/bin/python /local/wernerju/hannah/hannah/compression.py -i /local/wernerju/hannah/trained_models/test/tc-res8/last.ckpt

    file_path = args.filename
    ws, lengths, shapes, original_ws = load_parameters(file_path)

    print('----------- Replacement of Clusters by indices -------------')
    cluster = args.n_clusters
    ws_indexed, index_LUT = replace_cluster_by_indices(ws, cluster)

    print('----------- Huffman Encoding -------------')
    hs, tree = Huffman_encoding(ws_indexed)

    total_bits = calc_diff(hs)
    print('Number of required Bits in total: ', total_bits)
    print('Potential reduction: ', 1-(total_bits/(sum(lengths)*32)))


    print('----------- Huffman Decoding -------------')
    decoding = Huffman_decoding(hs, tree)
    norm = 0
    for i in range(len(decoding)):
        norm += np.linalg.norm(decoding[i]-ws_indexed[i])
    print('Norm of decoded weights and indexed weights: ', norm)
    ws_cluster, ws_tensor = rebuild_struc(decoding, index_LUT, ws, shapes, original_ws)

    m = torch.load(file_path, map_location='cpu')  # 'cpu' must be specified, otherwise a CUDA error can occur.
    idx = 0
    new_state_dict = dict()
    for k, v in m["state_dict"].items():
        if "weight" in k and "downsample" not in k:
            new_state_dict[k] = ws_tensor[idx]
            idx += 1


main()
