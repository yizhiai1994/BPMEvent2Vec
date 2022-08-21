# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:42:52 2019

@author: JGF
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW_at(nn.Module):
    def __init__(self,vocab_size,embed_size,window_size,output_size,batch,CUDA_type=False):
        super(CBOW_at, self).__init__()
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.output_size=output_size
        self.window_size=window_size
        self.batch=batch
        self.CUDA_type=CUDA_type
        
        
        self.embeddings=nn.Embedding(self.vocab_size,self.embed_size)
        self.liner1=nn.Linear(self.window_size*2*2*self.embed_size,self.output_size)
        self.liner2=nn.Linear(self.window_size*2*2*self.embed_size,self.output_size)

    def initembedding(self):
        initrange = 0.5 / self.embd_dimension
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        
    def forward(self,input_x):
        input_x=torch.tensor(input_x)
        embed_x=self.embeddings(input_x)
        #print(embed_x.shape)
        #print(embed_x.view(embed_x.shape[0],2,-1))
        #print(embed_x)
       
        embed_x=embed_x.view(embed_x.shape[0],self.window_size*2*2*self.embed_size)
        output_a=self.liner1(embed_x)
        output_t=self.liner2(embed_x)
        output_a=F.log_softmax(output_a,1)
        output_t=F.log_softmax(output_t,1)
        #print(output_t.size())
        output=torch.cat((output_a,output_t),1)
        #print(output.size())
        return output
    
    def save_embedding(self, id2event, vector_name):
        """Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.
        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        if self.CUDA_type:
            embedding = self.embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.embeddings.weight.data.numpy()
        vector = open(vector_name, 'w')
        vector.write('%d\t%d\n' % (len(id2event), self.embed_size))
        for wid, w in id2event.items():
            e = embedding[wid]
            e = '\t'.join(map(lambda x: str(x), e))
            vector.write('%s\t%s\n' % (w, e))
        vector.close()
        