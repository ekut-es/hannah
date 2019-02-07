from enum import Enum

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from ..utils import ConfigType, SerializableModule

class DNNSpeechModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        width = config["width"]
        height = config["height"]


        x = Variable(torch.zeros(1,1,width,height))
        
        self.dense = nn.ModuleList()

        
        x = x.view(1,-1)
        last_size = x.size(1)
        count = 1
        while "dnn{}_size".format(count) in config:
            dnn_size = config["dnn{}_size".format(count)]
            dense = nn.Linear(last_size, dnn_size)
            self.dense.append(dense)
            x = dense(x)
            
            relu = nn.ReLU()
            self.dense.append(relu)
            x = relu(x)

            dropout = nn.Dropout(config["dropout_prob"]) 
            self.dense.append(dropout)
            x = dropout(x)

            last_size = x.view(1, -1).size(1)
            
            count += 1

        self.output = nn.Linear(last_size, n_labels)
        x = self.output(x)
        last_size = x.view(1,-1).size(1)

        sum = 0
        for param in self.parameters():
            print(param.size())
            sum += param.view(-1).size(0)
    
        print("total_paramters:", sum)


    def forward(self, x):
        x = x.view(x.size(0),-1)
        for layer in self.dense:
            x = layer(x)

        x = self.output(x)
       
        return x



configs = {
    ConfigType.HELLO_DNN_SMALL.value : dict(
        dropout_prob=0.5,
        n_labels=12,
        dnn1_size = 144,
        dnn2_size = 144,
        dnn3_size = 144
    ),
    ConfigType.HELLO_DNN_MEDIUM.value : dict(
        dropout_prob=0.5,
        n_labels=12,
        dnn1_size = 256,
        dnn2_size = 256,
        dnn3_size = 256
    ),
    ConfigType.HELLO_DNN_LARGE.value : dict(
        dropout_prob=0.5,
        n_labels=12,
        dnn1_size = 436,
        dnn2_size = 436,
        dnn3_size = 436
    )
    
}
