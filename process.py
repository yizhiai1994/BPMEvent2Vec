# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:51:49 2019

@author: JGF
"""

import pandas as pd
f=open('.\data\BPI_Challenge_2017_extend.csv')
lines=f.readlines()
i=0
data=list()
for line in lines:
    if i==0:
        i+=1
        continue
    #print(line)
    temp=line.replace('\n','').split(',')
    event=temp[1].split('_')
    if event[1]!=event[-2]:
        act=event[0]+'-'+event[1]+'-'+event[-2]+'_'+event[-1]
    else:
        act=event[0]+'-'+event[1]+'_'+event[-1]
    data.append([temp[0],act,temp[2]])

data=pd.DataFrame(data,columns=['CaseID','Activity','CompleteTimestamp'])
data.to_csv('./data/BPI_Challenge_2017_extend(1).csv',index=False)
f.close()
print(1)