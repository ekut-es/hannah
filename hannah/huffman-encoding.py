import os.path
import onnx
from onnx import numpy_helper
import numpy as np
import matplotlib.pyplot as plt
import heapq


class node:
    def __init__(self, frq, left, right):
        self.frq = frq
        self.left = left
        self. right = right


def load_parameters(file_path):
    model = onnx.load(os.path.dirname(__file__) + file_path)
    parameters = []
    for i in model.graph.initializer:
        param = numpy_helper.to_array(i)
        parameters.append(param.flatten())
    return parameters



def get_frequencies(parameters):
    frequency = {}
    for matrix in parameters:
        for weight in matrix:
            if weight in frequency.keys():
                frequency[weight] += 1
            else:
                frequency[weight] = 1
        total = sum(frequency.values())
        frequency = {key: value / total for key, value in frequency.items()}
    return frequency



def enocode_Huffman(frq):
    nodes = []
    for char, frq in frq.items():
        heapq.heappush(nodes, (frq, char))
    while nodes:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        sum_frq = left[0] + right[0]

        #k = node(frq=sum_frq, left=left[1], right=right[1])
        #heapq.heappush(nodes, (sum_frq, ))
        #print(heapq.heappop(nodes))
    



def decode_Huffman():
    pass



def main():
    file_path = '/../trained_models/test/conv_net_trax/model.onnx'
    parameters = load_parameters(file_path)
    frq = get_frequencies(parameters)
    enocode_Huffman(frq)
  


if __name__ == "__main__":
    main()

#for j in parameters.values():
    #fig = plt.figure()
    #plt.hist(j.flatten(), bins=100)
    #plt.savefig('hist.png')
    #break

