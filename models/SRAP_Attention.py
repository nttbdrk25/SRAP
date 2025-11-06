import torch
import math
import torch.nn as nn
import torch.nn.functional as F 
from .common import *
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, activation='relu'):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = get_activation(activation) if relu else None
        #self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
def interleave(x1,x2):##x1 has the same dimesion as x2
    batch_size1, channels1, height1, width1 = x1.size()
    batch_size2, channels2, height2, width2 = x2.size()
    if channels1 == channels2:
        x = torch.stack((x1,x2), dim=2).view(batch_size1,2*channels1,height1,width1)
    elif channels1 > channels2:
        #print('vaoday: ' + str(channels1))
        temp_x1 = x1[:,0:channels2,:,:]
        temp_x2 = x1[:,channels2:channels1,:,:]
        x = torch.stack((temp_x1,x2), dim=2).view(batch_size2,2*channels2,height2,width2)
        x = torch.cat((x,temp_x2),1)
    elif channels2 > channels1:
        temp_x1 = x2[:,0:channels1,:,:]
        temp_x2 = x2[:,channels1:channels2,:,:]
        x = torch.stack((x1,temp_x1), dim=2).view(batch_size1,2*channels1,height1,width1)
        x = torch.cat((x,temp_x2),1)
    return x
def interleave3(x1,x2,x3):##x1 has the same dimesion as x2 and x3
    batch_size1, channels1, height1, width1 = x1.size()
    batch_size2, channels2, height2, width2 = x2.size()
    batch_size3, channels3, height3, width3 = x3.size()
    if channels1 == channels2 == channels3:
        x = torch.stack((x1,x2,x3), dim=2).view(batch_size1,3*channels1,height1,width1)
    else:
        min_channel = min(channels1,channels2,channels3)
        temp_x1 = x1[:,0:min_channel,:,:]
        temp_x2 = x2[:,0:min_channel,:,:]
        temp_x3 = x3[:,0:min_channel,:,:]
        x = torch.stack((temp_x1,temp_x2,temp_x3), dim=2).view(batch_size2,3*min_channel,height2,width2)
        x11 = x1[:,min_channel:channels1,:,:]
        x21 = x2[:,min_channel:channels2,:,:]
        x31 = x3[:,min_channel:channels3,:,:]
        if x11.nelement() != 0:
            x = torch.cat((x,x11),1)
        if x21.nelement() != 0:
            x = torch.cat((x,x21),1)
        if x31.nelement() != 0:
            x = torch.cat((x,x31),1)
    return x
def get_statistic_feature(x, pool_type=None):
    squeeze_channel = None
    squeeze_spatial = None        
    if pool_type=='avg':
        squeeze_channel = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        squeeze_spatial = torch.mean(x,1).unsqueeze(1)            
    if pool_type=='max':
        squeeze_channel = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))                
        squeeze_spatial = torch.max(x,1)[0].unsqueeze(1)
    if pool_type=='std':                
        stdf = torch.std(x,(2,3),unbiased=True)#compute standard deviation
        stdf = stdf.reshape(stdf.size()[0],stdf.size()[1],1,1)#resize to be (,1,1) the same as out put of AdaptiveAvgPool2d , i.e., self.squeeze(residual)
        squeeze_channel = stdf
        squeeze_spatial = torch.std(x,1).unsqueeze(1)
    return squeeze_channel, squeeze_spatial
def topk_channel(tensor_x,num):
    tensor_x = tensor_x.squeeze(-1).transpose(-1, -2)    
    toprate,indices1 = torch.topk(tensor_x,num)#get top of a half of x
    toprate = toprate.transpose(-1, -2).unsqueeze(-1)
    return toprate

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16,pool_types=['avg','max'],rate=None):
        super(ChannelGate, self).__init__()        
        #rate = 0.5
        self.pool_types = pool_types
        if len(pool_types)==3:
            self.num1 = int(rate[0]*in_channels)
            self.num2 = int(rate[1]*in_channels)
            self.num3 = in_channels - (self.num1 + self.num2)
        if len(pool_types)==2:
            self.num1 = int(rate[0]*in_channels)
            self.num2 = in_channels - self.num1
        #fix in_channels1<=reduction_ratio
        if in_channels // reduction_ratio == 0:
            reduction_ratio = in_channels
        #reduction_ratio = 1
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
            )
        self.in_channels = in_channels
        if len(self.pool_types)==3:
            self.spatialConv = BasicConv(3, 1, 7, stride=1, padding=3, relu=False)
        if len(self.pool_types)==2:
            self.spatialConv = BasicConv(2, 1, 7, stride=1, padding=3, relu=False)
    def forward(self, x):
        if len(self.pool_types)==3:
            squeeze_channel1, squeeze_spatial1 = get_statistic_feature(x,self.pool_types[0])
            squeeze_channel2, squeeze_spatial2 = get_statistic_feature(x,self.pool_types[1])
            squeeze_channel3, squeeze_spatial3 = get_statistic_feature(x,self.pool_types[2])
            
            squeeze_channel11 = topk_channel(squeeze_channel1,self.num1)        
            squeeze_channel22 = topk_channel(squeeze_channel2,self.num2)
            squeeze_channel33 = topk_channel(squeeze_channel3,self.num3)
            
            squeeze_all = interleave3(squeeze_channel11,squeeze_channel22,squeeze_channel33)
    
            squeeze_weight = F.relu(self.mlp(squeeze_all).unsqueeze(2).unsqueeze(3) + F.sigmoid(squeeze_channel1*squeeze_channel2*squeeze_channel3))#_residualSE(OK best)
            spatial_all = torch.cat((squeeze_spatial1, squeeze_spatial2,squeeze_spatial3), dim=1)
            spatial_weight = self.spatialConv(spatial_all) + F.sigmoid(squeeze_spatial1*squeeze_spatial2*squeeze_spatial3)
        if len(self.pool_types)==2:
            squeeze_channel1, squeeze_spatial1 = get_statistic_feature(x,self.pool_types[0])
            squeeze_channel2, squeeze_spatial2 = get_statistic_feature(x,self.pool_types[1])
            squeeze_channel11 = topk_channel(squeeze_channel1,self.num1)        
            squeeze_channel22 = topk_channel(squeeze_channel2,self.num2)
            squeeze_all = interleave(squeeze_channel11,squeeze_channel22)
            squeeze_weight = F.relu(self.mlp(squeeze_all).unsqueeze(2).unsqueeze(3) + F.sigmoid(squeeze_channel1*squeeze_channel2))#_residualSE(OK best)
            spatial_all = torch.cat((squeeze_spatial1, squeeze_spatial2), dim=1)                        
            spatial_weight = self.spatialConv(spatial_all) + F.sigmoid(squeeze_spatial1*squeeze_spatial2)
        
        weight_all = 1 + F.sigmoid(squeeze_weight * spatial_weight)            
        return x * weight_all
    
class SRAP(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16,pool_types=None,rate=None):
        super(SRAP, self).__init__()
        self.ChannelGate = ChannelGate(in_channels, reduction_ratio,pool_types=pool_types,rate=rate)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out
