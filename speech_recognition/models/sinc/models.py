from enum import Enum
import math
import sys
import os

from typing import Dict, Any


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import logging
msglogger = logging.getLogger()


import pwlf
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "distiller"))
print(sys.path)
import distiller


from ..utils import ConfigType, SerializableModule, next_power_of2

#Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
#https://github.com/mravanelli/SincNet

################################# Sinc Convolution ############################

class SincConv(nn.Module):
    """Sinc convolution:
        Parameters:
        -----------------
            in_channels: No. of input channels(must be 1)
            out_channels: No. of filters(40)
            SR: sampling rate, default set at 32000
            kernel_size: Filter length(101)
            """
    @staticmethod
    def to_mel(hz):
        return 2595*np.log10(1+hz/700)

    @staticmethod
    def to_hz(mel):
        return 700*(10**(mel/2595)-1)
    
    def __init__(self,out_channels,kernel_size,SR=16000,in_channels=1,stride=1,padding=0,dilation=1,bias=False,groups=1,min_low_hz=50,min_band_hz=50):
        super(SincConv,self).__init__()
        
        if in_channels!=1:
            err="SincConv only suports one input channel."
            raise ValueError(err)
            
        if bias:
            raise ValueError('SincConv does not support bias.')
        
        if groups>1:
            raise ValueError('SincConv only supports one group.')
        
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.SR=SR
        self.min_low_hz=min_low_hz
        self.min_band_hz=min_band_hz
        
        if kernel_size%2==0:
            kernel_size=kernel_size+1 #odd length so that filter is symmetric
        
        #initializing filter banks in audible frequency range
        low_hz=30
        high_hz=SR/2-(self.min_band_hz+self.min_low_hz)
        
        mel=np.linspace(self.to_mel(low_hz),self.to_mel(high_hz),self.out_channels+1)
        hz=self.to_hz(mel)
        
        self.low_freq_=nn.Parameter(torch.Tensor(hz[:-1]).view(-1,1)/self.SR)
        self.band_freq_=nn.Parameter(torch.Tensor(np.diff(hz)).view(-1,1)/self.SR)
        
        #hamming window         
        N=(self.kernel_size-1)/2.0
        self.window_=torch.hamming_window(self.kernel_size)
        #self.window_=0.54-0.46*torch.cos(2*math.pi*torch.linspace(1,N,steps=N)/self.kernel_size)
        self.n_=2*math.pi*torch.arange(-N,0).view(1,-1)
        
    def forward(self, waveforms):
            
        self.n_=self.n_.to(waveforms.device)
        self.window_=self.window_.to(waveforms.device)
            
        f_low=torch.abs(self.low_freq_)+min_low_hz
        f_high=f_low+min_band_hz+torch.abs(self.band_freq_)
        f_band=(f_high-f_low)[:,0]
            
        f_n_low=torch.matmul(f_low,self.n_)
        f_n_high=torch.matmul(f_high,self.n_)
            
        bpl=((torch.sin(f_n_high)-torch.sin(f_n_low))/(self.n_/2))
        bpr=torch.flip(bpl,dims=[1])
        bpc=2*f_band.view(-1,1)
            
        band=torch.cat([bpl,bpc,bpr],dim=1)
        band=band/(2*f_band[:,None])
        band=band*self.window_[None,]
        
        self.filters=band.view(self.out_channels,1,self.kernel_size)
            
        return F.conv1d(waveforms,self.filters,stride=self.stride,padding=self.padding,dilation=self.dilation,bias=None,groups=1)
        
########################## Activation Function ################################

def act_fun(input):
    # Returns the log compression value of the input
    return torch.log10(torch.abs(input)+1)

class my_act(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        
        return act_fun(input)
    
############################### SincConv Block ################################
    
class SincConvBlock(nn.Module):
    def __init__(self,N_filt,filt_len,bn_len,avgpool_len,SR,stride):
        super(SincConvBlock,self).__init__()
        #config will be the configuration file containing info about the architecture
        
        self.layer=nn.Sequential(
            SincConv(N_filt,filt_len,SR,stride=stride),
            my_act(),
            nn.BatchNorm1d(bn_len),
            nn.AvgPool1d(avgpool_len)
            )
        
    def forward(self,x):
        #batch=x.shape[0]
        #seq_len=x.shape[1]
        #x=x.view(batch,1,seq_len)
        
        out=self.layer(x)
        #out=out.view(batch,-1) Not very sure about this
        
        return out
    
############################# DS Conv Block ###################################
      
class GDSConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,groups,padding=0,dilation=1,bias=False):
        super(GDSConv,self).__init__()
        
        self.layer1=nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias) #depthwise convolution with k*1 filters
        self.layer2=nn.Conv1d(in_channels,out_channels,1,1,0,1,groups=groups,bias=bias)
        #pointwise convolutions with 1*c/g filters
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        
        return x

class GDSConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,groups,avg_pool_len,bn_len,spatDrop,padding=0,dilation=1,bias=False):
        super(GDSConvBlock,self).__init__()
        
        self.layer=nn.Sequential(
            GDSConv(in_channels,out_channels,kernel_size,stride,groups),
            nn.ReLU(),
            nn.BatchNorm1d(bn_len),
            nn.AvgPool1d(avg_pool_len),
            nn.Dropout(spatDrop)
            )
        
    def forward(self, x):
        x=self.layer(x)
        
        return x

############################ Combined Final Block *****************************

class FinalBlock(SerializableModule):
    def __init__(self,config):
        super(FinalBlock,self).__init__()	

        self.num_classes=config['num_classes']
		        
        self.cnn_N_filt=config['cnn_N_filt']
        self.cnn_filt_len=config['cnn_filt_len']
        self.cnn_bn_len=config['cnn_bn_len']
        self.cnn_avgpool_len=config['cnn_avgpool_len']
        self.SR=config['SR']
        self.cnn_stride=config['cnn_stride']
        
        self.dsconv_N_filt=config['dsconv_N_filt']
        self.dsconv_filt_len=config['dsconv_filt_len']
        self.dsconv_stride=config['dsconv_stride']
        self.dsconv_groups=config['dsconv_groups']
        self.dsconv_avg_pool_len=config['dsconv_avg_pool_len']
        self.dsconv_bn_len=config['dsconv_bn_len']
        self.dsconv_spatDrop=config['dsconv_spatDrop']
        self.dsconv_num=len(config['dsconv_N_filt'])
        
        self.SincNet=SincConvBlock(self.cnn_N_filt,self.cnn_filt_len,self.cnn_bn_len,self.cnn_avgpool_len,self.SR,self.cnn_stride)
        self.GDSBlocks=nn.ModuleList([])
        
        for i in range(self.dsconv_num):
            if i==0:
                self.GDSBlocks.append(GDSConvBlock(self.cnn_N_filt,self.dsconv_N_filt[i],self.dsconv_filt_len[i],self.dsconv_stride[i],self.dsconv_groups[i],self.dsconv_avg_pool_len[i],self.dsconv_bn_len[i],self.dsconv_spatDrop[i]))
                
            else:
                self.GDSBlocks.append(GDSConvBlock(self.dsconv_N_filt[i-1],self.dsconv_N_filt[i],self.dsconv_filt_len[i],self.dsconv_stride[i],self.dsconv_groups[i],self.dsconv_avg_pool_len[i],self.dsconv_bn_len[i],self.dsconv_spatDrop[i]))
                
        self.Global_avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.softmax_layer=nn.Softmax(self.num_classes)
    
    def forward(self,x):
        x=x.view(1,1,len(x))
        x=self.SincNet(x)
        
        for i in range(self.dsconv_num):
            x=self.GDSBlocks[i](x)
            
        x=self.Global_avg_pool(x)
        x=self.softmax_layer(x)
        
        return x

###############################################################################    


configs= {
	
	ConfigType.SINC.value: dict(
		num_classes=12,
		cnn_N_filt=40,
		cnn_filt_len=101,
		cnn_bn_len=40,
		cnn_avgpool_len=2,
		SR=16000,
		cnn_stride=8,

		dsconv_N_filt=(160,160,160,160,160),
		dsconv_filt_len=(25,9,9,9,9),
		dsconv_stride=(2,1,1,1,1),
		dsconv_groups=(1,2,4,2,4),
		dsconv_avg_pool_len=(2,2,2,2,2),
		dsconv_bn_len=(160,160,160,160,160),
		dsconv_spatDrop=(0.1,0.1,0.1,0.1,0.1),
	),

}
