from enum import Enum

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from ..utils import ConfigType, SerializableModule

class RawSpeechModel(SerializableModule):
    """Speech Recognition on RAW Data using Wolfgang Fuhls Networks"""
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        width = config["input_length"]
        
        last_size = 0

        x = Variable(torch.zeros(1,1,width))
        
        count = 1
        while "conv{}_size".format(count) in config:

            fmap_in_name     = "n_feature_maps_{}".format(count)
            fmap_out_name    = "n_feature_maps_{}".format(count+1)
            conv_size_name   = "conv{}_size".format(count)
            conv_stride_name = "conv{}_stride".format(count)

            pool_size_name = "pool{}_size".format(count)
            pool_stride_name = "pool{}_stride".format(count)
            
            if conv_size_name in config:
                n_feature_maps_in = config[fmap_in_name]
                n_feature_maps_out = config[fmap_out_name]
                conv_size = config[conv_size_name]
                conv_stride = config[conv_stride_name]
                pad = conv_size // 2
                conv = nn.Conv1d(n_feature_maps_in,
                                 n_feature_maps_out,
                                 conv_size,
                                 conv_stride,
                                 padding=pad)
                x = conv(x)
                self.add_module("conv{}".format(count), conv)

            last_size = x.view(1,-1).size(1)
            print("Last size:", "conv{}".format(count), last_size, x.size())

                
            if  pool_size_name in config:
                pool_size = config[pool_size_name]
                pool_stride = config[pool_stride_name]
                pool = nn.MaxPool1d(pool_size, pool_stride)
                x = pool(x)
                self.add_module("pool{}".format(count), pool)


            last_size = x.view(1,-1).size(1)
            print("Last size:", "pool{}".format(count), last_size, x.size())

            count += 1

        
        x = x.view(1,-1)
            
        count = 1
        while "dnn{}_size".format(count) in config:
            dnn_size = config["dnn{}_size".format(count)]

            dnn = nn.Linear(last_size, dnn_size)
            self.add_module("dnn{}".format(count), dnn)

            x = dnn(x)
            last_size = x.view(1,-1).size(1)
            print("Last size:", "dnn{}".format(count), last_size, x.size())

            count += 1
            
        self.output = nn.Linear(last_size, n_labels)
        x = self.output(x)
        last_size = x.view(1,-1).size(1)
        print("Last size:", "dnn{}".format(count), last_size, x.size())

        self.add_module("dnn_last", self.output)

        dropout_prob = config["dropout_prob"]
        self.add_module("dropout", nn.Dropout(dropout_prob))
        
    def forward(self, x):
        if hasattr(self, 'conv1'):
            x = self.conv1(x)
            x = self.dropout(x)
        if hasattr(self, 'pool1'):
            x = self.pool1(x)

        if hasattr(self, 'conv2'):
            x = self.conv2(x)
            x = self.dropout(x)

        if hasattr(self, 'pool2'):
            x = self.pool2(x)

        if hasattr(self, 'conv3'):
            x = self.conv3(x)
            x = self.dropout(x)

        if hasattr(self, 'pool3'):
            x = self.pool3(x)

        # Reshape tensor for Dense Layers
        x = x.view(x.size(0), -1)

        if hasattr(self, 'dnn1'):
            x = self.dnn1(x)
            x = self.dropout(x)

        if hasattr(self, 'dnn2'):
            x = self.dnn2(x)
            x = self.dropout(x)

        x = self.output(x)

        return x


configs= {
     ConfigType.EKUT_RAW_CNN3_1D.value: dict(
         preprocessing="raw",
        dropout_prob = 0.5,
        n_labels = 4,
        conv1_size = 21,
        conv1_stride = 5,
        pool1_size = 4,
        pool1_stride = 4, 
        n_feature_maps_1 = 1,
        conv2_size = 5,
        conv2_stride = 1,
        pool2_size = 4,
        pool2_stride = 4, 
        n_feature_maps_2 = 16,
        conv3_size = 21,
        conv3_stride = 1,
        pool3_size = 4,
        pool3_stride = 4, 
        n_feature_maps_3 = 32,
        n_feature_maps_4 = 64,
        dnn1_size = 256,
    )
}
