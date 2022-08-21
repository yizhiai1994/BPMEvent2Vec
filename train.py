# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:02:48 2019

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
from datetime import datetime
import os

def train(address,window_size,embed_size,hidden_size,batch_size=5,model_type='model',
          max_epcho_size = 500,min_lr = 0.0000001,CUDA_use=False):
    
    save_first_floder,save_sec_floder,dataset=getsavadir(address,model_type,window_size,embed_size,hidden_size)
    data=Input_data(address,batch_size,window_size)
    
    data.readfile()
    data.getdata()
    data.get_batch()
    #print(data.batch_temp[3])
    output_size=data.eventcout
    vocab_size=data.eventcout
    if model_type=='model':
        model=CBOW_at(vocab_size,embed_size,window_size,output_size,batch_size)
    else:
        print(1)
        model=CBOW_at1(vocab_size,embed_size,window_size,output_size,batch_size)
    #print(model)
    learn_rate = 0.01
    learn_rate_change = learn_rate * 0.1
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = opt.Adam(model.parameters(), lr=learn_rate)
    loss_deque = deque(maxlen=10)
    total=1
    epcho=0
    data_temp=data.get_event_batch()
    while data_temp!=False:
        total+=1
        data_temp=data.get_event_batch()
    print(2)
    model_count=0
    while learn_rate>=min_lr and epcho<max_epcho_size:
        
        #print(1)
        acc=0
        acc_a=0
        acc_t=0
        acc_total=0
        count=0
        epcho+=1
        data.copy_event_batch()
        data_temp=data.get_event_batch()
        total_loss=torch.FloatTensor([0])
    
        while data_temp!=False:
            (context, target) = data_temp
            if len(context) == 0:
                data_temp = data.get_event_batch()
                continue
            model.zero_grad()
            prediction = model(context)
            target = torch.FloatTensor(target)
            loss = loss_function(prediction, target)
            
            loss.backward()
            pre_targets=prediction.view(prediction.shape[0],2,-1).argmax(dim=2)
            targets=target.view(prediction.shape[0],2,-1).argmax(dim=2)
            t_acc,t_count,t_acc_a,t_acc_t,t_acc_tol=getacc(pre_targets,targets)
            acc+=t_acc
            count+=t_count
            acc_a+=t_acc_a
            acc_t+=t_acc_t
            acc_total+=t_acc_tol
            optimizer.step()
            total_loss += loss.data
            data_temp = data.get_event_batch()
            #loss_deque保存最近十轮训练的损失
        loss_deque.append(total_loss.item())
        #loss_change记录本次训练的损失与前十次损失的平均的差值的绝对值，目的是记录损失的变化，用于自动调整学习率
        #通过改变loss_change队列的大小可以决定损失稳定多少轮后改变学习率
        loss_change = total_loss.item() - sum(loss_deque)/10
        loss_change = abs(loss_change)
        #如果十轮之内比损失变化小于10，则改变学习率
        test_data_num = 0
        test_data_true = 0
        if loss_change < 1:
            #保存当前epoch训练状态
            model_count+=1
            #print(count)
            for i in range(len(save_sec_floder)):
                if i==0:
                    model_save=save_sec_floder[i]+dataset+'_lr_'+str(model_count)+'.pth'
                    #print(model_save)
                    torch.save(model,model_save)
                else:
                    vector_save=save_sec_floder[i]+dataset+'_lr_'+str(model_count)+'.txt'
                    #print(vector_save)
                    model.save_embedding(data.id2event,vector_save)  
                
                
            if learn_rate > learn_rate_change:
                learn_rate = learn_rate - learn_rate_change
            else:
                learn_rate_change = learn_rate_change * 0.1
                learn_rate = learn_rate - learn_rate_change
            
            optimizer = opt.Adam(model.parameters(), lr=learn_rate)
            loss_deque = deque(maxlen=10)
            loss_deque.append(total_loss.item())
        print('acc',acc*1.0/count)
        print('acc_a',acc_a*1.0/count)
        print('acc_t',acc_t*1.0/count)
        print('acc_total',acc_total*1.0/count)
        print('epoch: ',epcho)
        print('lr: ',learn_rate)
        print('total_loss ',loss_deque)

def getacc(pre_targets,targets):
    acc=0
    acc_a=0
    acc_t=0
    acc_tol=0
    count=0
    for pre,y in zip(pre_targets,targets):
    #print(pre[0].data.item())
        pre_a=pre[0].data.item()
        pre_t=pre[1].data.item()
        tar_a=y[0].data.item()
        tar_t=y[1].data.item()
        if pre_a==tar_a:
            acc+=1
            acc_a+=1
        if pre_t==tar_t:
            acc+=1
            acc_t+=1
        if pre_a==tar_a and pre_t==tar_t:
            acc_tol+=1
        count+=2
    return acc,count,acc_a,acc_t,acc_tol 

def getsavadir(address,model_type,windows_size,embd_dimension,hidden_size):
    dataset=address.replace('./data/','').replace('.csv','')
    
    #print(dataset)
    model_para=model_type+'windows' + str(windows_size) + '_vector' + \
                 str(embd_dimension) + '_hidden' + str(hidden_size)
    day_time=datetime.now().strftime('%Y-%m-%d')
    minute_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
    save_first_floder=['./model/','./vector/']
    save_sec_floder=[]
    
    for floder in save_first_floder:
        floder_temp=floder+dataset+'/'+model_para+'/'
        save_sec_floder.append(floder_temp)
        if not os.path.exists(floder_temp):
            os.makedirs(floder_temp)
        print(floder_temp)
            
    return save_first_floder,save_sec_floder,dataset
     
    

train('./data/BPI_Challenge_2017_extend.csv',1,10,20,5,'model')
#save_first_floder,save_sec_floder,dataset=getsavadir('./data/helpdesk_extend.csv',2,10,20)
    
