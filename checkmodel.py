# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022 

@author: tuann 
"""
from torchsummary import summary
from models.mobilenet_with_MRAP import *
#model = build_mobilenet_v1(120, width_multiplier=1, cifar=False,pool_types=['avg','std','max'],rate=[0.3,0.5,0.2])
model = build_mobilenet_v3(120, "large", width_multiplier=1, cifar=False, use_lightweight_head=False,pool_types=['avg','std'],rate=[0.5,0.5])
model = model.cuda()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
#summary(model, (3, 32, 32))

