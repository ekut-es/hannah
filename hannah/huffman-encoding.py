import os.path
import onnx
from onnx import numpy_helper
import numpy as np
import matplotlib.pyplot as plt
import heapq


class Node:
    def __init__(self, frq, char, left=None, right=None, huff=''):
        self.frq = frq
        self.char = char
        self.left = left
        self.right = right
        self.huff = huff


def load_parameters(file_path):
    model = onnx.load(os.path.dirname(__file__) + file_path)
    parameters = []
    for i in model.graph.initializer:
        param = numpy_helper.to_array(i)
        parameters.append(param.flatten())
    return parameters



def get_frequencies(parameters):
    frq = {}
    for matrix in parameters:
        for weight in matrix:
            if weight in frq.keys():
                frq[weight] += 1
            else:
                frq[weight] = 1
        total = sum(frq.values())
        frq = {key: value / total for key, value in frq.items()}
    return frq



def create_tree(frq):
    nodes = []
    for char, frq in frq.items():
        node = Node(frq, char)
        heapq.heappush(nodes, (node.frq, id(node), node))
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        sum_frq = left[0] + right[0]
        left[2].huff = '0'
        right[2].huff = '1'
        internal_node = Node(sum_frq, 0, left, right)
        heapq.heappush(nodes, (internal_node.frq, id(internal_node), internal_node))
    return nodes[0]
        
        

def encode_Huffman(tree, h, encoding):
    k = h + tree[2].huff
    if tree[2].left:
        encode_Huffman(tree[2].left, k, encoding)
    if tree[2].right:
        encode_Huffman(tree[2].right, k, encoding)
    else:
        encoding[tree[2].char] = k
    return encoding


def decode_Huffman():
    pass



def main():
    file_path = '/../trained_models/test/conv_net_trax/model.onnx'
    parameters = load_parameters(file_path)

    frq = get_frequencies(parameters)
    tree = create_tree(frq)

    encoding = {}
    huffman_encoding = encode_Huffman(tree, '', encoding)

    params = [None] * len(parameters)
    for i in range(len(parameters)):
        params[i] = [huffman_encoding[code] for code in parameters[i]]
    print(params)


if __name__ == "__main__":
    main()

#for j in parameters.values():
    #fig = plt.figure()
    #plt.hist(j.flatten(), bins=100)
    #plt.savefig('hist.png')
    #break

