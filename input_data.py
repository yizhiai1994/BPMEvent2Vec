# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:06:41 2019

@author: JGF
"""

import numpy as np
from sklearn.model_selection import train_test_split

class Input_data:
    def __init__(self,data_address,batch_size,window_size=2):
        self.data_address=data_address
        self.orgin_data=[]
        self.event2id=dict()
        self.id2event=dict()
        self.eventcout=0
        self.data=[]
        self.batch_size=batch_size
        self.window_size=window_size
        self.evet_batch=list()
        self.batch_temp=list()
        self.at_batch=list()
        self.at_temp=list()
        
        #self.data_temp=[]
        
    def readfile(self):
        f=open(self.data_address)
        lines=f.readlines()
        i=0 
        flag=lines[1].replace('\r','').replace('\n','').split(',')[0]
        #print(flag)
        trace_temp=list()
        
        for t in range(11):
            self.event2id.setdefault('t_'+str(t),self.eventcout)
            self.id2event.setdefault(self.eventcout,'t_'+str(t))
            self.eventcout+=1
        self.event2id.setdefault('None',self.eventcout)
        self.id2event.setdefault(self.eventcout,'None')
        self.eventcout+=1
        for line in lines:
            
            if i==0:
                i+=1
                continue
            line=line.replace('\r','').replace('\n','').split(',')
            
            #print(flag==line[0])
            if flag==line[0]:
                #print(flag,' ',line[0])
                a,t=line[1].split('_')
                #print(a,t)
                
                trace_temp.append([a,t])
                
            else:
                flag=line[0]
                self.orgin_data.append(trace_temp.copy())
                trace_temp=list()
                a,t=line[1].split('_')
                #print(a,t)
                
                
                trace_temp.append([a,t])
            if a not in self.event2id.keys():
                #print(a)
                self.event2id.setdefault(a,self.eventcout)
                self.id2event.setdefault(self.eventcout,a)
                self.eventcout+=1
            
                
                
    def getdata(self):
        for trace in self.orgin_data:
            trace_temp=list()
            for event in trace:
                if event[1]!='None':
                    trace_temp.append([self.event2id[event[0]],self.event2id['t_'+event[1]]])
                else:
                    trace_temp.append([self.event2id[event[0]],self.event2id[event[1]]])
            self.data.append(trace_temp.copy())
        #self.data_temp=self.data.copy()
    
    def get_batch(self):
        
        input_temp=list()
        target_temp=list()
        input_final=list()
        i=1
        at_temp=list()
        for line in self.data:
            
            target_pos=self.window_size
            target_left=0
            target_right=self.window_size*2
            #print(target_right,len(line))
            while target_right<len(line):
                a_temp=list()
                t_temp=list()
                at_final=list()
                temp=list()
                tar_temp=list()
                input_final=list()
                temp=line[target_left:target_pos] + line[target_pos+1:target_right + 1]
                
                tar_temp=line[target_pos].copy()
                target1=[0]*self.eventcout
                target2=[0]*self.eventcout
                t1,t2=tar_temp
                target1[t1]=1
                target2[t2]=1
                target=target1+target2
                target_temp.append(target.copy())
                
                for event in temp:
                    #print(event)
                    input_final.append(event[0])
                    input_final.append(event[1])
                    a_temp.append(event[0])
                    t_temp.append(event[1])
                at_final=a_temp+t_temp
                at_temp.append(at_final.copy())
                #print(at_temp)
                input_temp.append(input_final.copy())
                #print(input_temp)
                #print(input_final)
                #print(1)
                #print(target_temp)
                #print(3)
                if len(input_temp)==self.batch_size:
                    self.batch_temp.append([input_temp.copy(),target_temp.copy()])
                    self.at_temp.append([at_temp.copy(),target_temp.copy()])
                    input_temp=list()
                    target_temp=list()
                    at_temp=list()
                    
                target_pos+=1
                target_left+=1
                target_right+=1
        i+=1
        
        if len(input_temp) != self.batch_size:
            #print(1)
            self.batch_temp.append([input_temp.copy(), target_temp.copy()])
            self.at_temp.append([at_temp.copy(),target_temp.copy()])
            #break
        self.event_batch = self.batch_temp.copy()
        self.at_batch=self.at_temp.copy()
            
    def get_event_batch(self):
        if len(self.batch_temp) != 0:
            return self.batch_temp.pop()
        return False
    
    def get_event_batch1(self):
        if len(self.at_temp) != 0:
            return self.at_temp.pop()
        return False
    
    def copy_event_batch(self):
        self.batch_temp = self.event_batch.copy()
        self.at_batch=self.at_temp.copy()
        
        
    def create_teain_test_split(self,threshold):
        self.train,self.test = train_test_split(self.event_batch,test_size=threshold)

'''
data=Input_data('./data/BPI_Challenge1_2015_extend.csv',5)
data.readfile()
print(data.orgin_data[:1])
data.getdata()
data.get_batch()
print(data.batch_size)
print(data.at_batch[:2])
'''
                