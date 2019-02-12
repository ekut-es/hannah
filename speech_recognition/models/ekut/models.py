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

        self.convolutions = nn.ModuleList()
        self.dense = nn.ModuleList()

        count = 1
        while "conv{}_size".format(count) in config:

            fmap_in_name     = "n_feature_maps_{}".format(count)
            fmap_out_name    = "n_feature_maps_{}".format(count+1)
            conv_size_name   = "conv{}_size".format(count)
            conv_stride_name = "conv{}_stride".format(count)
            conv_dilation_name = "conv{}_dilation".format(count)

            pool_size_name = "pool{}_size".format(count)
            pool_stride_name = "pool{}_stride".format(count)
            
            if conv_size_name in config:
                n_feature_maps_in = config[fmap_in_name]
                n_feature_maps_out = config[fmap_out_name]
                conv_size = config[conv_size_name]
                conv_stride = config[conv_stride_name]
                conv_dilation = 1

                if conv_dilation_name in config:
                    conv_dilation = config[conv_dilation_name]

                pad = conv_size * conv_dilation // 2
                conv = nn.Conv1d(n_feature_maps_in,
                                 n_feature_maps_out,
                                 conv_size,
                                 conv_stride,
                                 padding=pad,
                                 dilation=conv_dilation)
                x = conv(x)
                self.convolutions.append(conv)

                activation = nn.ReLU()
                self.convolutions.append(activation)
                x = activation(x)


                dropout = nn.Dropout(config["dropout_prob"])
                self.convolutions.append(dropout)
                x = dropout(x)
                
            last_size = x.view(1,-1).size(1)
            print("Last size:", "conv{}".format(count), last_size, x.size())

                
            if  pool_size_name in config:
                pool_size = config[pool_size_name]
                pool_stride = config[pool_stride_name]
                pool = nn.MaxPool1d(pool_size, pool_stride)
                x = pool(x)
                self.convolutions.append(pool)


            last_size = x.view(1,-1).size(1)
            print("Last size:", "pool{}".format(count), last_size, x.size())

            count += 1

        
        x = x.view(1,-1)
            
        count = 1
        while "dnn{}_size".format(count) in config:
            dnn_size = config["dnn{}_size".format(count)]

            dnn = nn.Linear(last_size, dnn_size)
            self.dense.append(dnn)

            x = dnn(x)
            last_size = x.view(1,-1).size(1)
            print("Last size:", "dnn{}".format(count), last_size, x.size())

            count += 1

            activation = nn.ReLU()
            self.dense.append(activation)
            x = activation(x)


            dropout = nn.Dropout(config["dropout_prob"])
            self.dense.append(dropout)
            x = dropout(x)
                

        self.output = nn.Linear(last_size, n_labels)
        x = self.output(x)
        last_size = x.view(1,-1).size(1)
        print("Last size:", "dnn{}".format(count), last_size, x.size())

        sum = 0
        for param in self.parameters():
            sum += param.view(-1).size(0)
    
        print("total_paramters:", sum)
        
    def forward(self, x, export=False):
        

        if export:
            f = open("layer_outputs.h", "w")


        if export:
            data = x.detach().numpy().flatten()
            f.write("static fp_t input[] = {" + ",".join((str(x) for x in data)) + "};\n\n")

            
        num = 0
        for layer in self.convolutions:

            if type(layer) == nn.modules.dropout.Dropout:
                continue
            
            x = layer(x)

            if export:
                data = x.detach().numpy().flatten()
                f.write("static fp_t output_layer" + str(num) + "[] = {" + ",".join((str(x) for x in data)) + "};\n\n")

            
            num += 1
            
        x = x.view(x.size(0),-1)
        for layer in self.dense:            
            x = layer(x)

            if type(layer) == nn.modules.dropout.Dropout:
                continue
            
            if export:
                data = x.detach().numpy().flatten()
                f.write("static fp_t output_layer" + str(num) + "[] = {" + ",".join((str(x) for x in data)) + "};\n\n")

            num += 1
            
        x = self.output(x)

        if export:
            data = x.detach().numpy().flatten()
            f.write("static fp_t output_layer" + str(num) + "[] = {" + ",".join((str(x) for x in data)) + "};\n\n")
        
        return x


configs= {
     ConfigType.EKUT_RAW_CNN2_1D.value: dict(
        features="raw",
        dropout_prob = 0.5,
        n_labels = 4,
        n_feature_maps_1 = 1,
        conv1_size = 21,
        conv1_stride = 5,
        pool1_size = 4,
        pool1_stride = 4, 
        n_feature_maps_2 = 16,
        conv2_size = 5,
        conv2_stride = 1,
        pool2_size = 4,
        pool2_stride = 4,
        n_feature_maps_3 = 32,
        conv3_size = 3,
        conv3_stride = 1,
        n_feature_maps_4 = 5,
        dnn1_size = 256,
    ),
    ConfigType.EKUT_RAW_CNN3_1D.value: dict(
        features="raw",
        dropout_prob = 0.5,
        n_labels = 4,
        n_feature_maps_1 = 1,
        conv1_size = 21,
        conv1_stride = 5,
        pool1_size = 4,
        pool1_stride = 4, 
        n_feature_maps_2 = 16,
        conv2_size = 5,
        conv2_stride = 1,
        pool2_size = 4,
        pool2_stride = 4,
        n_feature_maps_3 = 32,
        conv3_size = 5,
        conv3_stride = 1,
        pool3_size = 4,
        pool3_stride = 4, 
        n_feature_maps_4 = 64,
        dnn1_size = 256,
    ),

    ConfigType.EKUT_RAW_CNN3_1D_NARROW.value: dict(
        preprocessing="raw",
        dropout_prob = 0.5,
        n_labels = 4,
        n_feature_maps_1 = 1,
        conv1_size = 5,
        conv1_stride = 1,
        pool1_size = 4,
        pool1_stride = 4, 
        n_feature_maps_2 = 8,
        conv2_size = 5,
        conv2_stride = 1,
        pool2_size = 4,
        pool2_stride = 4,
        n_feature_maps_3 = 16,
        conv3_size = 5,
        conv3_stride = 1,
        pool3_size = 4,
        pool3_stride = 4, 
        n_feature_maps_4 = 32,
        dnn1_size = 128,
    )
    
}
