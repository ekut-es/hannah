import os
import onnx
import numpy as np
from onnx import numpy_helper
import copy
import torch
from huffman import Huffman_encoding, Huffman_decoding

def load_parameters(file_path):
    model = onnx.load(os.path.dirname(__file__) + file_path)
    parameters = []
    for i in model.graph.initializer:
        if "weight" in i.name: # for conv_net_trax, add other case for non_quantized models
            param = numpy_helper.to_array(i)
            parameters.append(param.flatten())
    return parameters



def replace_cluster_by_indices(parameters):
    ws = copy.deepcopy(parameters)
    cluster = 10  # number of clusters
    ws_indexed = []
    print(len(parameters[0]))
    index_LUT = np.full(shape=(len(ws), cluster+1), fill_value=0) 
    print(index_LUT)
    for k in range(len(ws)):
        centers = np.unique(ws[k], return_counts=False)  # get unique cluster centers
        # print(len(centers))
        '''for j in range(len(centers)):  # fill LUT
            index_LUT[k,j] = centers[j]'''
        intermediate_layer = np.select([ws[k]==centers[j] for j in range(len(centers))], 
                    [i+1 for i in range(len(centers))], 
                    ws[k])  # replace with index 1 to i for i cluster
        ws_indexed.append(intermediate_layer)
    return ws_indexed


def main():
    file_path = '/../trained_models/test/tc-res8/model.onnx'
    parameters = load_parameters(file_path) 

    print('----------- Replacement of Clusters by indices -------------')
    ws_indexed = replace_cluster_by_indices(parameters)

    print('----------- Huffman Encoding -------------')
    hs, tree = Huffman_encoding(ws_indexed)

    print('----------- Huffman Decoding -------------')
    decoding = Huffman_decoding(hs, tree)
    print('Norm of decoded weights and indexed weights: ', (np.linalg.norm(torch.FloatTensor(decoding)==ws_indexed))) # check if indexed ws and decoded sequence are equal
    

main()


'''# Replace centroids by indices, 0. added through pading in linear layer results in one more cluster 
ws_indexed, index_LUT = replace_cluster_by_indices(length_layers, ws)    
print('---------- Cluster centers were replaced with indices}------------')
         
# Huffman encoding
print('----------- Huffman Encoding -------------')
hs, tree = Huffman_encoding(ws_indexed)
        
# Huffman decoding and replacement of indices by cluster centers
decoding = Huffman_decoding(hs, tree)
print((torch.FloatTensor(decoding)==ws_indexed).all()) # check if indexed ws and decoded sequence are equal
ws_cluster = replace_indices_by_clusters(length_layers, ws_indexed, index_LUT, ws)

        
counter = 0
total_bits = 0
for i in range(len(hs)):
    total_bits += len(hs[i])
    if len(hs[i]) > 384:
        counter += 1
print('Number of rows exceeding 384 Bit: ', counter)
print('Number of required Bits in total: ', total_bits)
print('Potential reduction: ', 1-(total_bits/(384*len(hs))))'''