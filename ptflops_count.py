# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:25:35 2023 

@author: tuann  
"""
import torch
from ptflops import get_model_complexity_info
from torchsummary import summary
from models.mobilenet_with_MRAP import *
with torch.cuda.device(0):        
  model = build_mobilenet_v2(120, width_multiplier=1, cifar=False,pool_types=['avg','std'],rate=[0.5,0.5])  
  #model = build_mobilenet_v3(120, "large", width_multiplier=1, cifar=False, use_lightweight_head=False,pool_types=['avg','std'],rate=[0.5,0.5])
  #macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
  macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True,
                                           #flops_units='MMac')
                                           flops_units='GMac')
  print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
  macs1 = macs.split()
  strmacs1=str(float(macs1[0])/2) + ' ' + macs1[1][0]
  print('{:<30}  {:<8}'.format('Floating-point operations (FLOPs): ', strmacs1))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  print('Number of model parameters (referred)): {}'.format(
      sum([p.data.nelement() for p in model.parameters()])))
  #summary(model, (3, 224, 224))

