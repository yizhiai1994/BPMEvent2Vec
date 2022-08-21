# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:56:27 2019

@author: JGF
"""

import torch 
import numpy as np
from model import CBOW_at
from input_data import Input_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from collections import deque
from model1 import CBOW_at1

data=Input_data('./data/helpdesk_extend.csv',5,2)
    
data.readfile()
data.getdata()
data.get_batch()
    #print(data.batch_temp[3])
output_size=data.eventcout
vocab_size=data.eventcout
model=CBOW_at1(vocab_size,10,2,output_size,5)
input_x,target=data.get_event_batch1()
print(input_x)
#print(target)

prediction=model(input_x)

'''
target=torch.tensor(target)
pre_targets=prediction.view(prediction.shape[0],2,-1).argmax(dim=2)
targets=target.view(prediction.shape[0],2,-1).argmax(dim=2)
acc=0

#print(targets)
print(pre_targets)
print(targets)

for pre,y in zip(pre_targets,targets):
    #print(pre[0].data.item())
    if pre[0].data.item()==y[0].data.item():
        acc+=1
    if pre[1].data.item()==y[1].data.item():
        acc+=1
print(acc)
#print(pre)
'''