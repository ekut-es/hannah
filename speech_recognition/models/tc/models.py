from enum import Enum
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from ..utils import ConfigType, SerializableModule

class TCResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, size, stride, dilation, clipping_value,  bottleneck, channel_division, separable):
        super().__init__()
        self.stride = stride
        self.clipping_value = clipping_value
        if stride > 1:
            # No dilation needed: 1x1 kernel
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.Hardtanh(0.0, self.clipping_value))
        
        pad_x = size[0] // 2
        pad_y = size[1] // 2

        if bottleneck:
            groups = output_channels//channel_division if separable else 1
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, output_channels//channel_division, (1,1), stride=1, dilation=dilation, bias=False),
                nn.Conv2d(output_channels//channel_division, output_channels//channel_division, size, stride=stride, padding=(dilation*pad_x,dilation*pad_y), dilation=dilation, bias=False, groups=groups),
                nn.Conv2d(output_channels//channel_division, output_channels, (1,1), stride=1, dilation=dilation, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.Hardtanh(0.0, self.clipping_value),
                nn.Conv2d(output_channels, output_channels//channel_division, (1,1), stride=1, dilation=dilation, bias=False),
                nn.Conv2d(output_channels//channel_division, output_channels//channel_division, size, 1, padding=(dilation*pad_x,dilation*pad_y), dilation=dilation, bias=False, groups=groups),
                nn.Conv2d(output_channels//channel_division, output_channels, (1,1), stride=1, dilation=dilation, bias=False),
                nn.BatchNorm2d(output_channels))
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, size, stride, padding=(dilation*pad_x,dilation*pad_y), dilation=dilation, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.Hardtanh(0.0, self.clipping_value),
                nn.Conv2d(output_channels, output_channels, size, 1, padding=(dilation*pad_x,dilation*pad_y), dilation=dilation, bias=False),
                nn.BatchNorm2d(output_channels))

            
        self.relu = nn.Hardtanh(0.0, self.clipping_value)
            
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
        dilation = config["dilation"]        
        clipping_value = config["clipping_value"]
        bottleneck = config["bottleneck"]
        channel_division = config["channel_division"]
        separable = config["separable"]

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

                # Change first convolution to bottleneck layer.
                if bottleneck[0] == 1:
                    channel_division_local = channel_division[0]
                    # Change bottleneck layer to sepearable convolution
                    groups =  output_channels//channel_division_local if separable[0] else 1
            
                    conv1 = nn.Conv2d(input_channels, output_channels//channel_division_local, (1,1), 1, bias = False)
                    conv2 = nn.Conv2d(output_channels//channel_division_local, output_channels//channel_division_local, size, stride, bias = False, groups=groups)
                    conv3 = nn.Conv2d(output_channels//channel_division_local, output_channels, (1,1), 1, bias = False)
                    self.layers.append(conv1)
                    self.layers.append(conv2)
                    self.layers.append(conv3)
                else:
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
                
                # Use same bottleneck, channel_division factor and separable configuration for all blocks
                block = TCResidualBlock(input_channels, output_channels, size, stride, dilation ** count, clipping_value, bottleneck[1], channel_division[1], separable[1])
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
        
        return x
                        
        
  
configs= {
     ConfigType.TC_RES_8.value: dict(
        features="mel",
        fully_convolutional=False,
        dropout_prob = 0.5,
        width_multiplier = 1,
        dilation = 1,
        clipping_value = 100000,
        bottleneck = (0,0),
        channel_division = (2,4),
        separable = (0,0),
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
        fully_convolutional=False,
        width_multiplier = 1,
        dilation = 1,
        clipping_value = 100000,
        bottleneck = (0,0),
        channel_division = (4,2),
        separable = (0,0),
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
        fully_convolutional=False,
        width_multiplier = 1.5,
        dilation = 1,
        clipping_value = 100000,
        bottleneck = (0,0),
        channel_division = (4,2),
        separable = (0,0),
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
        width_multiplier = 1.5,
        dilation = 1,
        clipping_value = 100000,
        bottleneck = (0,0),
        channel_division = (4,2),
        separable = (0,0),
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
