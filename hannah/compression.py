import os
import onnx
import numpy as np
from onnx import numpy_helper
file_path = '/../trained_models/test/conv_net_trax/model.onnx'
model = onnx.load(os.path.dirname(__file__) + file_path)
[print(t.name) for t in model.graph.initializer]
m_graph = model.graph.initializer
weights = numpy_helper.to_array(m_graph[0])




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