import numpy as np
import heapq


class Node:
    def __init__(self, frq, char, left=None, right=None, huff=''):
        self.frq = frq
        self.char = char
        self.left = left
        self.right = right
        self.huff = huff


def get_frequencies(ws):
    frq = {} 
    for i in range(len(ws)):
        for j in range(len(ws[i])):
            if ws[i][j] in frq.keys():
                frq[ws[i][j]] += 1
            else:
                frq[ws[i][j]] = 1
    total = sum(frq.values())
    frq = {key: value / total for key, value in frq.items()} # relative frequencies
    return frq



def create_tree(frq):
    nodes = []
    for char, frq in frq.items():
        node = Node(frq, char)
        heapq.heappush(nodes, (node.frq, id(node), node)) # each character is a leaf node
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        sum_frq = left[0] + right[0] # join nodes of least frq
        left[2].huff = '0' # left edge
        right[2].huff = '1' #right edge
        internal_node = Node(sum_frq, 0, left, right) 
        heapq.heappush(nodes, (internal_node.frq, id(internal_node), internal_node)) # add new node into tree
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


def decode_Sequence(seq, tree):
    d = []
    root = tree
    for i in range(len(seq)):
        if seq[i] == '0' and root[2].left: # bit = 0, go left
            root = root[2].left
        elif root[2].right: # bit = 1, go right
            root = root[2].right
        if root[2].left is None and root[2].right is None: # leaf node
            d.append(root[2].char) 
            root = tree
    return d



def encode_Sequence(ws, huffman_dict):
    k = np.asarray(ws)
    params = [None] * len(k)
    ws = []
    for i in range(len(k)):
        encoded_array = [huffman_dict[w] for w in k[i]]
        params[i] = encoded_array    
    for j in range(len(params)):
        ws.append(''.join(params[j]))

    return ws


def Huffman_encoding(ws):
    frq = get_frequencies(ws)
    print('Huffman dictionary with relative frequencies as values, indices as keys: ', frq)
    tree = create_tree(frq)

    encoding = {}
    huffman_dict = encode_Huffman(tree, '', encoding)
    huffman_encoding = encode_Sequence(ws, huffman_dict)
    return huffman_encoding, tree


def Huffman_decoding(huffman_encoding, tree):
    decoding = []
    for i in range(len(huffman_encoding)):
        decoding.append(decode_Sequence(huffman_encoding[i], tree))
    return decoding

