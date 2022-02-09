from cmath import nan
from operator import index
import os
import onnx
import numpy as np
from onnx import numpy_helper
import copy
import torch
from huffman import Huffman_encoding, Huffman_decoding

def load_parameters(file_path):
    values = []
    lengths = []
    clustered_model = torch.load(os.path.dirname(__file__) + file_path)
    for key, value in clustered_model["state_dict"].items():
        if "weight" in key and "downsample" not in key:
            lengths.append(len(value.numpy().flatten()))
            values.append(value.numpy().flatten())
    return values, lengths



def replace_cluster_by_indices(parameters):
    ws = copy.deepcopy(parameters)
    cluster = 10  # number of clusters
    ws_indexed = []
    index_LUT = np.full(shape=(len(ws), cluster+1), fill_value=nan) 
    for k in range(len(ws)):
        centers = np.unique(ws[k], return_counts=False)  # get unique cluster centers
        for j in range(len(centers)):  # fill LUT
            index_LUT[k,j] = centers[j]
        intermediate_layer = np.select([ws[k]==centers[j] for j in range(len(centers))], 
                    [i+1 for i in range(len(centers))], 
                    ws[k])  # replace with index 1 to i for i cluster
        ws_indexed.append(intermediate_layer)
    return ws_indexed, index_LUT


def replace_indices_by_clusters(ws_indexed, index_LUT, ws):
    ws_indexed = copy.deepcopy(np.asarray(ws_indexed, dtype=object))
    ws_cluster = []
    for i in range(len(ws_indexed)):
        #print(ws_indexed[i])
        #print(ws_cluster[i])
        #print([ws_indexed[i]==x for x in range(1, index_LUT.shape[1]+1)])
        #print(np.select([ws_indexed[i]==x for x in range(1, index_LUT.shape[1]+1)],
        #[index_LUT[i,k] for k in range(index_LUT.shape[1])], ws_indexed[i])
        intermediate_layer = (np.select([ws_indexed[i]==x for x in range(1, index_LUT.shape[1]+1)],
        [index_LUT[i,k] for k in range(index_LUT.shape[1])], ws_indexed[i]))
        #print([ws_indexed[i] for x in range(1, index_LUT.shape[1]+1)])
    #print('Original weights are equal to decoded weights: ', (torch.FloatTensor(ws_cluster)==ws).all())
    #print('Normx of ws_cluster - ws: ', np.linalg.norm(torch.FloatTensor(ws_cluster)-ws))
    return ws_cluster

def calc_diff(hs):
    total_bits = 0
    for i in range(len(hs)):
        total_bits += len(hs[i])
    return total_bits
    


def main():
    file_path = '/../trained_models/test/tc-res8/last.ckpt'
    ws, lengths = load_parameters(file_path)

    print('----------- Replacement of Clusters by indices -------------')
    ws_indexed, index_LUT = replace_cluster_by_indices(ws)

    print('----------- Huffman Encoding -------------')
    hs, tree = Huffman_encoding(ws_indexed)

    print('----------- Huffman Decoding -------------')
    decoding = Huffman_decoding(hs, tree)
    norm = 0
    for i in range(len(decoding)):
        norm += np.linalg.norm(decoding[i]-ws_indexed[i])
    print('Norm of decoded weights and indexed weights: ', norm)
    #ws_cluster = replace_indices_by_clusters(decoding, index_LUT, ws) 

    total_bits = calc_diff(hs)
    print('Number of required Bits in total: ', total_bits)
    print('Potential reduction: ', 1-(total_bits/(sum(lengths)*32)))


main()
