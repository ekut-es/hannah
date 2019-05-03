from enum import Enum
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from ..utils import ConfigType, SerializableModule

class TCResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, size, stride):
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU())
        
        pad_x = size[0] // 2
        pad_y = size[1] // 2
        
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, size, stride, padding=(pad_x,pad_y), bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, size, 1, padding=(pad_x,pad_y), bias=False),
            nn.BatchNorm2d(output_channels))
            
        self.relu = nn.ReLU()
            
    def forward(self, x):
        y = self.convs(x)
        if self.stride > 1:
            x = self.downsample(x)
            
        res = self.relu(y + x)
        
        return res
  
class TCResNetModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        
        n_labels = config["n_labels"]
        width = config["width"]
        height = config["height"]
        dropout_prob = config["dropout_prob"]
        width_multiplier = config["width_multiplier"]
        self.fully_convolutional = config["fully_convolutional"]
        
        self.layers = nn.ModuleList()
  
        input_channels = height
  
        x = Variable(torch.zeros(1,height,width, 1))      
  
        count = 1
        while "conv{}_size".format(count) in config:
                output_channels_name = "conv{}_output_channels".format(count)
                size_name = "conv{}_size".format(count)
                stride_name = "conv{}_stride".format(count)
                
                output_channels = int(config[output_channels_name] * width_multiplier)
                size = config[size_name]
                stride = config[stride_name] 
                
                conv = nn.Conv2d(input_channels, output_channels, size, stride, bias = False)
                self.layers.append(conv)
                
                input_channels = output_channels
                count += 1
        
        count = 1
        while "block{}_conv_size".format(count) in config:
                output_channels_name = "block{}_output_channels".format(count)
                size_name = "block{}_conv_size".format(count)
                stride_name = "block{}_stride".format(count)
                
                output_channels = int(config[output_channels_name] * width_multiplier)
                size = config[size_name]
                stride = config[stride_name] 
                
                block = TCResidualBlock(input_channels, output_channels, size, stride)
                self.layers.append(block)
                
                input_channels = output_channels
                count += 1
        
        for layer in self.layers: 
            x = layer(x)
        
        shape = x.shape
        average_pooling = nn.AvgPool2d((shape[2], shape[3]))
        self.layers.append(average_pooling)
        
        x = average_pooling(x)

        if not self.fully_convolutional:
            x = x.view(1,-1)
            
        shape = x.shape
        self.dropout = nn.Dropout(dropout_prob)

        if self.fully_convolutional:
            self.fc = nn.Conv2d(shape[1], n_labels, 1, bias = False)
        else:
            self.fc = nn.Linear(shape[1], n_labels, bias=False)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0,2,3,1)
        for layer in self.layers:
            x = layer(x)
        
        x = self.dropout(x)
        if not self.fully_convolutional:
            x = x.view(x.size(0), -1)
        x = self.fc(x)

        print(x.shape)
        
        return x
                        
        
  
configs= {
     ConfigType.TC_RES_8.value: dict(
        features="mel",
        fully_convolutional=False,
        dropout_prob = 0.5,
        n_labels = 12,
        width_multiplier = 1,
        conv1_size = (3,1),
        conv1_stride = 1,
        conv1_output_channels = 16,
        block1_conv_size = (9,1),
        block1_stride = 2,
        block1_output_channels = 24,
        block2_conv_size = (9,1),
        block2_stride = 2,
        block2_output_channels = 32,
        block3_conv_size = (9,1),
        block3_stride = 2,
        block3_output_channels = 48 
    ),
    ConfigType.TC_RES_14.value: dict(
        features="mel",
        dropout_prob = 0.5,
        n_labels = 12,
        fully_convolutional=False,
        width_multiplier = 1,
        conv1_size = (3,1),
        conv1_stride = 1,
        conv1_output_channels = 16,
        block1_conv_size = (9,1),
        block1_stride = 2,
        block1_output_channels = 24,
        block2_conv_size = (9,1),
        block2_stride = 1,
        block2_output_channels = 24,
        block3_conv_size = (9,1),
        block3_stride = 2,
        block3_output_channels = 32,
        block4_conv_size = (9,1),
        block4_stride = 1,
        block4_output_channels = 32,
        block5_conv_size = (9,1),
        block5_stride = 2,
        block5_output_channels = 48,
        block6_conv_size = (9,1),
        block6_stride = 1,
        block6_output_channels = 48 
    ),
    ConfigType.TC_RES_8_15.value: dict(
        features="mel",
        dropout_prob = 0.5,
        n_labels = 12,
        fully_convolutional=False,
        width_multiplier = 1.5,
        conv1_size = (3,1),
        conv1_stride = 1,
        conv1_output_channels = 16,
        block1_conv_size = (9,1),
        block1_stride = 2,
        block1_output_channels = 24,
        block2_conv_size = (9,1),
        block2_stride = 2,
        block2_output_channels = 32,
        block3_conv_size = (9,1),
        block3_stride = 2,
        block3_output_channels = 48 
    ),
    ConfigType.TC_RES_14_15.value: dict(
        features="mel",
        dropout_prob = 0.5,
        fully_convolutional=False,
        n_labels = 12,
        width_multiplier = 1.5,
        conv1_size = (3,1),
        conv1_stride = 1,
        conv1_output_channels = 16,
        block1_conv_size = (9,1),
        block1_stride = 2,
        block1_output_channels = 24,
        block2_conv_size = (9,1),
        block2_stride = 1,
        block2_output_channels = 24,
        block3_conv_size = (9,1),
        block3_stride = 2,
        block3_output_channels = 32,
        block4_conv_size = (9,1),
        block4_stride = 1,
        block4_output_channels = 32,
        block5_conv_size = (9,1),
        block5_stride = 2,
        block5_output_channels = 48,
        block6_conv_size = (9,1),
        block6_stride = 1,
        block6_output_channels = 48 
    )
}
