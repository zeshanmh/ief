import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math

class MultiHeadedAttention(nn.Module):
    ''' 
    Taken from Sasha Rush's blogpost: 
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    '''

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h   = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)]) 
        self.attn    = None
        if dropout is not None: 
            self.dropout = nn.Dropout(p=dropout)
        else: 
            self.dropout = None

    def forward(self, query, key, value, mask=None, debug=False, use_matmul=False):
        "Implements Figure 2"
        '''
            If use_matmul is false (i.e. you want to do to featurewise attention), 
            then we expect the following shapes: 
                key: [bs, T-1, num IEFs, D]
                val: [bs, T-1, num IEFs, D]
                query: [bs, T-1, 1, D]
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        if not use_matmul: 
            query, key, value = \
                [l(x).view(nbatches,x.shape[1],x.shape[2],self.h,self.d_k).transpose(2,3).transpose(1,2) if len(x.shape) == 4 
                 else l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                 for l, x in zip(self.linears, (query, key, value))]
        else: 
            query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        if debug: 
            print(f'query shape: {query.shape}')
            print(f'key   shape: {key.shape}')
            print(f'value shape: {key.shape}')
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout, use_matmul=use_matmul)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None, use_matmul=True):
        d_k    = query.size(-1)
        if use_matmul: 
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            p_attn = F.softmax(scores, dim = -1)
            if dropout is not None:
                p_attn = dropout(p_attn)
            return torch.matmul(p_attn, value), p_attn
        else:
            # print('not using matmul...')
            scores = (query.transpose(-2,-1)*key.transpose(-2,-1)) / math.sqrt(d_k)
            # if mask is not None: 
            #     scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim = -1)
            if dropout is not None: 
                p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.transpose(-2,-1), dim = -1), p_attn
